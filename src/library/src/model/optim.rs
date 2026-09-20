use crate::{Axis, Module, ParamId, Parameter, Result, Tensor, backend::Buffer};
use std::collections::{HashMap, HashSet};

pub struct SGD {
    learning_rate: f32,
}

/// Device-resident Adam with bias correction and one state pair per ParamId.
pub struct Adam {
    learning_rate: f32,
    axis_learning_rates: Option<(Axis, Vec<f32>)>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: i32,
    states: HashMap<ParamId, AdamState>,
}

struct AdamState {
    first: Buffer,
    second: Buffer,
    learning_rates: Option<Buffer>,
}

struct AdamUpdate {
    parameter: Parameter,
    id: ParamId,
    tensor: Tensor,
    first: Buffer,
    second: Buffer,
    learning_rates: Option<Buffer>,
}

struct AdamPrepared {
    next_step: i32,
    updates: Vec<AdamUpdate>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Result<Self> {
        Self::with_hyperparameters(learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Adam with one learning rate per member of a named parameter axis.
    pub fn with_axis_learning_rates(axis: Axis, learning_rates: Vec<f32>) -> Result<Self> {
        if learning_rates.is_empty()
            || learning_rates
                .iter()
                .any(|rate| !rate.is_finite() || *rate <= 0.0)
        {
            return Err("Adam axis learning rates must be nonempty, finite, and positive".into());
        }
        let mut optimizer = Self::new(1.0)?;
        optimizer.axis_learning_rates = Some((axis, learning_rates));
        Ok(optimizer)
    }

    pub fn with_hyperparameters(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("Adam learning rate must be finite and positive".into());
        }
        if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
            return Err("Adam beta1 must be finite and within 0..1".into());
        }
        if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
            return Err("Adam beta2 must be finite and within 0..1".into());
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Adam epsilon must be finite and positive".into());
        }
        Ok(Self {
            learning_rate,
            axis_learning_rates: None,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            step: 0,
            states: HashMap::new(),
        })
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.step_parameters(model.parameters())
    }

    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        let prepared = self.prepare(parameters)?;
        self.commit(prepared);
        Ok(())
    }

    fn prepare(&self, parameters: impl IntoIterator<Item = Parameter>) -> Result<AdamPrepared> {
        let next_step = self.step.checked_add(1).ok_or("Adam step overflow")?;
        let correction1 = 1.0 - self.beta1.powi(next_step);
        let correction2 = 1.0 - self.beta2.powi(next_step);
        let mut seen = HashSet::new();
        let mut updates = vec![];
        for parameter in parameters {
            let id = parameter.id();
            if seen.insert(id) {
                let state = self.states.get(&id);
                let learning_rates = match (&self.axis_learning_rates, state) {
                    (_, Some(state)) => state.learning_rates.clone(),
                    (Some((axis, values)), None) => {
                        Some(parameter.tensor().expanded_axis_values(*axis, values)?)
                    }
                    (None, None) => None,
                };
                let (tensor, first, second) = parameter.tensor().adam_updated(
                    state.map(|state| &state.first),
                    state.map(|state| &state.second),
                    self.learning_rate,
                    learning_rates.as_ref(),
                    self.beta1,
                    self.beta2,
                    correction1,
                    correction2,
                    self.epsilon,
                    self.weight_decay,
                )?;
                updates.push(AdamUpdate {
                    parameter,
                    id,
                    tensor,
                    first,
                    second,
                    learning_rates,
                });
            }
        }
        Ok(AdamPrepared { next_step, updates })
    }

    fn commit(&mut self, prepared: AdamPrepared) {
        for update in prepared.updates {
            update.parameter.replace(update.tensor);
            self.states.insert(
                update.id,
                AdamState {
                    first: update.first,
                    second: update.second,
                    learning_rates: update.learning_rates,
                },
            );
        }
        self.step = prepared.next_step;
    }

    pub fn completed_steps(&self) -> i32 {
        self.step
    }
}

/// Adam with decoupled weight decay; moments and updates remain device-resident.
pub struct AdamW(Adam);

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err("AdamW weight decay must be finite and nonnegative".into());
        }
        let mut adam = Adam::new(learning_rate)?;
        adam.weight_decay = weight_decay;
        Ok(Self(adam))
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.0.step(model)
    }

    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        self.0.step_parameters(parameters)
    }

    pub fn completed_steps(&self) -> i32 {
        self.0.completed_steps()
    }
}

impl SGD {
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        Ok(Self { learning_rate })
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.step_parameters(model.parameters())
    }

    /// All uses accumulate into one leaf; each shared ParamId is updated exactly once.
    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let mut updates = vec![];
        for parameter in parameters {
            if seen.insert(parameter.id()) {
                let new = parameter.tensor().updated(self.learning_rate)?;
                updates.push((parameter, new));
            }
        }
        for (parameter, tensor) in updates {
            parameter.replace(tensor);
        }
        Ok(())
    }
}

const NEWTON_SCHULZ_A: f32 = 3.4445;
const NEWTON_SCHULZ_B: f32 = -4.7750;
const NEWTON_SCHULZ_C: f32 = 2.0315;
const NEWTON_SCHULZ_STEPS: usize = 5;
const NORMALIZATION_EPSILON: f32 = 1e-7;

/// The storage orientation of one explicitly selected Muon matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuonMatrixOrientation {
    /// Axis [`crate::Linear`] storage: `[fan_in, fan_out]`.
    FanInFanOut,
    /// Conventional PyTorch linear storage: `[fan_out, fan_in]`.
    FanOutFanIn,
}

/// A parameter selected for Muon together with its matrix orientation.
///
/// Muon is intended for hidden rank-2 weights. Embeddings, output heads,
/// biases, gains, and other parameters should normally use the auxiliary
/// AdamW arm of [`MuonWithAuxAdamW`]. Selection is explicit: Axis never picks
/// parameters merely because they happen to have rank two.
#[derive(Clone)]
pub struct MuonMatrix {
    id: ParamId,
    orientation: MuonMatrixOrientation,
}

impl MuonMatrix {
    /// Select an Axis [`crate::Linear`] weight stored as `[fan_in, fan_out]`.
    pub fn axis_linear(parameter: &Parameter) -> Self {
        Self::fan_in_fan_out(parameter)
    }

    /// Select a matrix stored as `[fan_in, fan_out]`.
    pub fn fan_in_fan_out(parameter: &Parameter) -> Self {
        Self {
            id: parameter.id(),
            orientation: MuonMatrixOrientation::FanInFanOut,
        }
    }

    /// Select a matrix stored as `[fan_out, fan_in]`.
    pub fn fan_out_fan_in(parameter: &Parameter) -> Self {
        Self {
            id: parameter.id(),
            orientation: MuonMatrixOrientation::FanOutFanIn,
        }
    }

    pub fn param_id(&self) -> ParamId {
        self.id
    }

    pub fn orientation(&self) -> MuonMatrixOrientation {
        self.orientation
    }
}

type MuonSelection = HashMap<ParamId, MuonMatrixOrientation>;

fn collect_muon_selection(selected: impl IntoIterator<Item = MuonMatrix>) -> Result<MuonSelection> {
    let mut selection = HashMap::new();
    for matrix in selected {
        if let Some(previous) = selection.insert(matrix.id, matrix.orientation)
            && previous != matrix.orientation
        {
            return Err(format!(
                "Muon parameter {:?} was selected with conflicting matrix orientations",
                matrix.id
            )
            .into());
        }
    }
    if selection.is_empty() {
        return Err("Muon requires at least one explicit matrix selection".into());
    }
    Ok(selection)
}

struct SelectedMuonParameter {
    parameter: Parameter,
    orientation: MuonMatrixOrientation,
}

/// Momentum updates orthogonalized by five Newton-Schulz iterations.
///
/// Axis follows the pinned
/// [Muon reference implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py);
/// its copyright and MIT notice ship in the crate's `THIRD_PARTY.md`.
///
/// The selected matrices must cover every unique parameter in the model passed
/// to [`Muon::step`]. Use [`MuonWithAuxAdamW`] when the model also contains
/// biases, gains, heads, embeddings, or other auxiliary parameters.
///
/// Missing gradients are errors and leave all parameters, optimizer state, and
/// step counters unchanged. The defaults are momentum `0.95`, Nesterov enabled,
/// five Newton-Schulz steps, and FP32 optimizer state. On a BF16 device only
/// matrix-product inputs are rounded to BF16; accumulation and state stay FP32.
pub struct Muon {
    selected: MuonSelection,
    learning_rate: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    step: i32,
    states: HashMap<ParamId, Tensor>,
}

struct MuonUpdate {
    parameter: Parameter,
    id: ParamId,
    tensor: Tensor,
    momentum: Tensor,
}

struct MuonPrepared {
    next_step: i32,
    updates: Vec<MuonUpdate>,
}

impl Muon {
    pub fn new(
        selected: impl IntoIterator<Item = MuonMatrix>,
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<Self> {
        Self::with_hyperparameters(selected, learning_rate, 0.95, true, weight_decay)
    }

    pub fn with_hyperparameters(
        selected: impl IntoIterator<Item = MuonMatrix>,
        learning_rate: f32,
        momentum: f32,
        nesterov: bool,
        weight_decay: f32,
    ) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("Muon learning rate must be finite and positive".into());
        }
        if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
            return Err("Muon momentum must be finite and within 0..1".into());
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err("Muon weight decay must be finite and nonnegative".into());
        }
        Ok(Self {
            selected: collect_muon_selection(selected)?,
            learning_rate,
            momentum,
            nesterov,
            weight_decay,
            step: 0,
            states: HashMap::new(),
        })
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        let (parameters, auxiliary) = partition_parameters(model, &self.selected)?;
        if !auxiliary.is_empty() {
            return Err(format!(
                "Muon selection omits {} unique model parameter(s); use MuonWithAuxAdamW for an auxiliary remainder",
                auxiliary.len()
            )
            .into());
        }
        let prepared = self.prepare(parameters)?;
        self.commit(prepared);
        Ok(())
    }

    fn prepare(
        &self,
        parameters: impl IntoIterator<Item = SelectedMuonParameter>,
    ) -> Result<MuonPrepared> {
        let next_step = self.step.checked_add(1).ok_or("Muon step overflow")?;
        let mut unique = vec![];
        for selected in parameters {
            let parameter = &selected.parameter;
            if parameter.tensor().shape().rank() != 2 {
                return Err(format!(
                    "Muon parameter {:?} must have rank 2; observed rank {}",
                    parameter.id(),
                    parameter.tensor().shape().rank()
                )
                .into());
            }
            if parameter.grad().is_none() {
                return Err(format!(
                    "Muon parameter {:?} has no gradient; run backward before step",
                    parameter.id()
                )
                .into());
            }
            unique.push(selected);
        }
        if unique.is_empty() {
            return Err("Muon requires at least one selected rank-2 parameter".into());
        }

        let mut updates = Vec::with_capacity(unique.len());
        for selected in unique {
            let parameter = selected.parameter;
            let id = parameter.id();
            let gradient = parameter.grad().expect("validated gradient");
            let observation_weight = 1.0 - self.momentum;
            let next_momentum = match self.states.get(&id) {
                Some(previous) => previous
                    .scale(self.momentum)?
                    .add(&gradient.scale(observation_weight)?)?,
                None => gradient.scale(observation_weight)?,
            };
            let direction = if self.nesterov {
                gradient
                    .scale(observation_weight)?
                    .add(&next_momentum.scale(self.momentum)?)?
            } else {
                next_momentum.clone()
            };
            let orthogonal = zeroth_power(&direction)?;
            let parameter_tensor = parameter.tensor();
            let dims = parameter_tensor.shape().dims();
            let (fan_in, fan_out) = match selected.orientation {
                MuonMatrixOrientation::FanInFanOut => (dims[0].extent, dims[1].extent),
                MuonMatrixOrientation::FanOutFanIn => (dims[1].extent, dims[0].extent),
            };
            let scaling = (fan_out as f32 / fan_in as f32).max(1.0).sqrt();
            let decayed = parameter_tensor
                .detach()
                .scale(1.0 - self.learning_rate * self.weight_decay)?;
            let tensor = decayed.sub(&orthogonal.scale(self.learning_rate * scaling)?)?;
            updates.push(MuonUpdate {
                parameter,
                id,
                tensor,
                momentum: next_momentum,
            });
        }
        Ok(MuonPrepared { next_step, updates })
    }

    fn commit(&mut self, prepared: MuonPrepared) {
        for update in prepared.updates {
            update.parameter.replace(update.tensor);
            self.states.insert(update.id, update.momentum);
        }
        self.step = prepared.next_step;
    }

    pub fn completed_steps(&self) -> i32 {
        self.step
    }
}

fn partition_parameters(
    model: &impl Module,
    selected: &MuonSelection,
) -> Result<(Vec<SelectedMuonParameter>, Vec<Parameter>)> {
    let mut seen = HashSet::new();
    let mut found = HashSet::new();
    let mut muon = vec![];
    let mut auxiliary = vec![];
    for parameter in model.parameters() {
        let id = parameter.id();
        if !seen.insert(id) {
            continue;
        }
        if let Some(&orientation) = selected.get(&id) {
            found.insert(id);
            muon.push(SelectedMuonParameter {
                parameter,
                orientation,
            });
        } else {
            auxiliary.push(parameter);
        }
    }
    if found.len() != selected.len() {
        let mut missing: Vec<_> = selected.keys().filter(|id| !found.contains(id)).collect();
        missing.sort_by_key(|id| format!("{id:?}"));
        return Err(format!(
            "Muon selection contains {} ParamId(s) that do not belong to this model: {missing:?}",
            missing.len()
        )
        .into());
    }
    Ok((muon, auxiliary))
}

fn zeroth_power(matrix: &Tensor) -> Result<Tensor> {
    let dims = matrix.shape().dims();
    debug_assert_eq!(dims.len(), 2);
    let row = dims[0].axis;
    let column = dims[1].axis;
    let (small, large) = if dims[0].extent <= dims[1].extent {
        (row, column)
    } else {
        (column, row)
    };
    let inner = small.role("muon_ns_inner");
    let output = small.role("muon_ns_output");
    let mut x = matrix.normalized_l2(NORMALIZATION_EPSILON)?;
    for _ in 0..NEWTON_SCHULZ_STEPS {
        let gram = x.contract(&x.rename(small, inner)?, large)?;
        let gram_rhs = gram.rename(inner, output)?.rename(small, inner)?;
        let gram_squared = gram.contract(&gram_rhs, inner)?.rename(output, inner)?;
        let b = gram
            .scale(NEWTON_SCHULZ_B)?
            .add(&gram_squared.scale(NEWTON_SCHULZ_C)?)?;
        let bx = b.contract(&x.rename(small, inner)?, inner)?;
        x = x.scale(NEWTON_SCHULZ_A)?.add(&bx)?;
    }
    Ok(x)
}

/// Explicitly routes selected rank-2 parameters to Muon and the exact
/// remainder to AdamW. No shape-based selection occurs.
pub struct MuonWithAuxAdamW {
    muon: Muon,
    adamw: AdamW,
}

impl MuonWithAuxAdamW {
    pub fn new(
        selected: impl IntoIterator<Item = MuonMatrix>,
        muon_learning_rate: f32,
        adamw_learning_rate: f32,
        weight_decay: f32,
    ) -> Result<Self> {
        Ok(Self {
            muon: Muon::new(selected, muon_learning_rate, weight_decay)?,
            adamw: AdamW::new(adamw_learning_rate, weight_decay)?,
        })
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        let (muon_parameters, auxiliary_parameters) =
            partition_parameters(model, &self.muon.selected)?;

        // Prepare both arms before either optimizer mutates parameters or state.
        let muon = self.muon.prepare(muon_parameters)?;
        let adamw = self.adamw.0.prepare(auxiliary_parameters)?;
        self.muon.commit(muon);
        self.adamw.0.commit(adamw);
        Ok(())
    }

    pub fn completed_steps(&self) -> i32 {
        debug_assert_eq!(self.muon.completed_steps(), self.adamw.completed_steps());
        self.muon.completed_steps()
    }

    pub fn muon(&self) -> &Muon {
        &self.muon
    }

    pub fn adamw(&self) -> &AdamW {
        &self.adamw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Shape};

    #[test]
    fn muon_hyperparameters_reject_invalid_values() -> Result<()> {
        let empty = || std::iter::empty::<MuonMatrix>();
        assert!(Muon::new(empty(), 0.0, 0.0).is_err());
        assert!(Muon::new(empty(), f32::NAN, 0.0).is_err());
        assert!(Muon::new(empty(), 0.02, f32::INFINITY).is_err());
        assert!(Muon::with_hyperparameters(empty(), 0.02, 1.0, true, 0.0).is_err());
        assert!(Muon::with_hyperparameters(empty(), 0.02, f32::NAN, false, 0.0).is_err());
        assert!(MuonWithAuxAdamW::new(empty(), 0.02, 0.001, 0.0).is_err());
        assert!(MuonWithAuxAdamW::new(empty(), 0.02, f32::NAN, 0.0).is_err());
        assert!(AdamW::new(f32::NAN, 0.0).is_err());
        assert!(AdamW::new(0.001, f32::INFINITY).is_err());
        Ok(())
    }

    fn close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance + tolerance * expected.abs(),
                "{label}[{index}]: actual={actual}, expected={expected}, error={error}"
            );
        }
    }

    fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
        let mut result = vec![0.0; values.len()];
        for row in 0..rows {
            for column in 0..columns {
                result[column * rows + row] = values[row * columns + column];
            }
        }
        result
    }

    fn matmul(left: &[f32], right: &[f32], rows: usize, inner: usize, columns: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                result[row * columns + column] = (0..inner)
                    .map(|k| left[row * inner + k] * right[k * columns + column])
                    .sum();
            }
        }
        result
    }

    fn scalar_zeroth_power(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
        // Intentionally literal: this oracle does not reuse implementation
        // constants, helpers, axis contractions, or orientation logic.
        let (alpha, beta, gamma) = (3.4445_f32, -4.7750_f32, 2.0315_f32);
        let transposed = rows > columns;
        let (mut x, small, large) = if transposed {
            (transpose(values, rows, columns), columns, rows)
        } else {
            (values.to_vec(), rows, columns)
        };
        let norm = x.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in &mut x {
            *value /= norm + 1e-7_f32;
        }
        for _ in 0..5 {
            let xt = transpose(&x, small, large);
            let gram = matmul(&x, &xt, small, large, small);
            let gram_squared = matmul(&gram, &gram, small, small, small);
            let polynomial: Vec<_> = gram
                .iter()
                .zip(&gram_squared)
                .map(|(matrix, squared)| beta * matrix + gamma * squared)
                .collect();
            let product = matmul(&polynomial, &x, small, small, large);
            x = x
                .iter()
                .zip(product)
                .map(|(x, product)| alpha * x + product)
                .collect();
        }
        if transposed {
            transpose(&x, columns, rows)
        } else {
            x
        }
    }

    #[derive(Clone, Copy)]
    struct ScalarMuon {
        learning_rate: f32,
        momentum: f32,
        nesterov: bool,
        weight_decay: f32,
    }

    fn scalar_muon_step(
        values: &mut [f32],
        momentum_buffer: &mut [f32],
        gradient: &[f32],
        shape: (usize, usize),
        orientation: MuonMatrixOrientation,
        optimizer: ScalarMuon,
    ) {
        let (rows, columns) = shape;
        let observation_weight = 1.0 - optimizer.momentum;
        for (buffer, gradient) in momentum_buffer.iter_mut().zip(gradient) {
            *buffer = optimizer.momentum * *buffer + observation_weight * gradient;
        }
        let direction = if optimizer.nesterov {
            gradient
                .iter()
                .zip(&*momentum_buffer)
                .map(|(gradient, buffer)| {
                    observation_weight * gradient + optimizer.momentum * buffer
                })
                .collect()
        } else {
            momentum_buffer.to_vec()
        };
        let update = scalar_zeroth_power(&direction, rows, columns);
        let (fan_in, fan_out) = match orientation {
            MuonMatrixOrientation::FanInFanOut => (rows, columns),
            MuonMatrixOrientation::FanOutFanIn => (columns, rows),
        };
        let scaling = (fan_out as f32 / fan_in as f32).max(1.0).sqrt();
        for (value, update) in values.iter_mut().zip(update) {
            *value = (1.0 - optimizer.learning_rate * optimizer.weight_decay) * *value
                - optimizer.learning_rate * scaling * update;
        }
    }

    fn set_gradient(parameter: &Parameter, gradient: &[f32]) -> Result<()> {
        let tensor = parameter.tensor();
        let coefficient: Vec<_> = gradient
            .iter()
            .map(|value| *value * tensor.shape().len() as f32)
            .collect();
        tensor
            .mul(&Tensor::from_slice(
                &coefficient,
                tensor.shape().dims().iter().copied(),
                tensor.device(),
            )?)?
            .mean(tensor.shape().axes())?
            .backward()
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn muon_matches_tall_and_wide_scalar_oracles_for_two_steps() -> Result<()> {
        let device = Device::cuda(0)?;
        for (rows, columns, orientation) in [
            (3, 2, MuonMatrixOrientation::FanInFanOut),
            (2, 3, MuonMatrixOrientation::FanInFanOut),
            (3, 2, MuonMatrixOrientation::FanOutFanIn),
        ] {
            let (row, column) = (Axis::new("row"), Axis::new("column"));
            let initial: Vec<_> = (0..rows * columns)
                .map(|index| (index as f32 - 2.0) * 0.11)
                .collect();
            let gradients = [
                (0..rows * columns)
                    .map(|index| (index as f32 + 1.0) * 0.07 - 0.19)
                    .collect::<Vec<_>>(),
                (0..rows * columns)
                    .map(|index| 0.23 - index as f32 * 0.035)
                    .collect::<Vec<_>>(),
            ];
            let parameter = Parameter::new(Tensor::from_slice(
                &initial,
                [row.of(rows), column.of(columns)],
                &device,
            )?);
            let matrix = match orientation {
                MuonMatrixOrientation::FanInFanOut => MuonMatrix::fan_in_fan_out(&parameter),
                MuonMatrixOrientation::FanOutFanIn => MuonMatrix::fan_out_fan_in(&parameter),
            };
            let mut model = Parameters(vec![
                ("weight".into(), parameter.clone()),
                ("tied_weight".into(), parameter.clone()),
            ]);
            let mut optimizer = Muon::with_hyperparameters([matrix], 0.04, 0.8, true, 0.03)?;
            let mut expected = initial;
            let mut momentum = vec![0.0; rows * columns];

            for (step, gradient) in gradients.iter().enumerate() {
                parameter.zero_grad();
                set_gradient(&parameter, gradient)?;
                optimizer.step(&mut model)?;
                scalar_muon_step(
                    &mut expected,
                    &mut momentum,
                    gradient,
                    (rows, columns),
                    orientation,
                    ScalarMuon {
                        learning_rate: 0.04,
                        momentum: 0.8,
                        nesterov: true,
                        weight_decay: 0.03,
                    },
                );
                // Orthogonalization removes a global rescaling of the search
                // direction. Check the state itself so the source EMA
                // recurrence cannot masquerade as unnormalized momentum; the
                // distinct second gradient exercises accumulated state.
                close(
                    &format!(
                        "{rows}x{columns} {orientation:?} Muon momentum step {}",
                        step + 1
                    ),
                    &optimizer
                        .states
                        .get(&parameter.id())
                        .expect("Muon state committed")
                        .to_vec()?,
                    &momentum,
                    2e-6,
                );
                close(
                    &format!("{rows}x{columns} {orientation:?} Muon step {}", step + 1),
                    &parameter.tensor().to_vec()?,
                    &expected,
                    3e-4,
                );
                assert_eq!(optimizer.completed_steps(), i32::try_from(step + 1)?);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn muon_without_nesterov_matches_two_gradient_scalar_oracle() -> Result<()> {
        let device = Device::cuda(0)?;
        let (row, column) = (Axis::new("row"), Axis::new("column"));
        let initial = [0.2, -0.1, 0.3, 0.5];
        let gradients = [[0.3, -0.2, 0.1, 0.4], [-0.12, 0.31, 0.08, -0.27]];
        let parameter = Parameter::new(Tensor::from_slice(
            &initial,
            [row.of(2), column.of(2)],
            &device,
        )?);
        let mut model = Parameters(vec![("weight".into(), parameter.clone())]);
        let mut optimizer = Muon::with_hyperparameters(
            [MuonMatrix::axis_linear(&parameter)],
            0.02,
            0.7,
            false,
            0.01,
        )?;
        let mut expected = initial.to_vec();
        let mut momentum = vec![0.0; 4];
        for (step, gradient) in gradients.iter().enumerate() {
            parameter.zero_grad();
            set_gradient(&parameter, gradient)?;
            optimizer.step(&mut model)?;
            scalar_muon_step(
                &mut expected,
                &mut momentum,
                gradient,
                (2, 2),
                MuonMatrixOrientation::FanInFanOut,
                ScalarMuon {
                    learning_rate: 0.02,
                    momentum: 0.7,
                    nesterov: false,
                    weight_decay: 0.01,
                },
            );
            close(
                &format!("Muon without Nesterov state step {}", step + 1),
                &optimizer
                    .states
                    .get(&parameter.id())
                    .expect("Muon state committed")
                    .to_vec()?,
                &momentum,
                2e-6,
            );
            close(
                &format!("Muon without Nesterov value step {}", step + 1),
                &parameter.tensor().to_vec()?,
                &expected,
                3e-4,
            );
        }
        Ok(())
    }

    struct Parameters(Vec<(String, Parameter)>);

    impl Module for Parameters {
        fn output_shape(&self, input: &Shape) -> Result<Shape> {
            Ok(input.clone())
        }

        fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
            Ok(input.clone())
        }

        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }

        fn named_parameters(&self) -> Vec<(String, Parameter)> {
            self.0.clone()
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn hybrid_partitions_exactly_deduplicates_ties_and_is_atomic() -> Result<()> {
        let device = Device::cuda(0)?;
        let (row, column, bias_axis) = (Axis::new("row"), Axis::new("column"), Axis::new("bias"));
        let weight = Parameter::new(Tensor::from_slice(
            &[0.2, -0.1, 0.3, 0.5],
            [row.of(2), column.of(2)],
            &device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &[0.4, -0.6],
            [bias_axis.of(2)],
            &device,
        )?);
        let mut model = Parameters(vec![
            ("weight".into(), weight.clone()),
            ("tied_weight".into(), weight.clone()),
            ("bias".into(), bias.clone()),
            ("tied_bias".into(), bias.clone()),
        ]);
        let weight_gradient = [0.3, -0.2, 0.1, 0.4];
        let bias_gradient = [0.25, -0.5];
        set_gradient(&weight, &weight_gradient)?;
        set_gradient(&bias, &bias_gradient)?;
        let mut optimizer = MuonWithAuxAdamW::new(
            [
                MuonMatrix::axis_linear(&weight),
                MuonMatrix::axis_linear(&weight),
            ],
            0.02,
            0.01,
            0.05,
        )?;
        optimizer.step(&mut model)?;
        assert_eq!(optimizer.completed_steps(), 1);

        let mut expected_weight = vec![0.2, -0.1, 0.3, 0.5];
        let mut momentum = vec![0.0; 4];
        scalar_muon_step(
            &mut expected_weight,
            &mut momentum,
            &weight_gradient,
            (2, 2),
            MuonMatrixOrientation::FanInFanOut,
            ScalarMuon {
                learning_rate: 0.02,
                momentum: 0.95,
                nesterov: true,
                weight_decay: 0.05,
            },
        );
        close(
            "hybrid Muon arm",
            &weight.tensor().to_vec()?,
            &expected_weight,
            3e-4,
        );
        let expected_bias: Vec<_> = [0.4_f32, -0.6]
            .into_iter()
            .zip(bias_gradient)
            .map(|(value, gradient)| {
                value - 0.01 * (gradient / (gradient.abs() + 1e-8) + 0.05 * value)
            })
            .collect();
        close(
            "hybrid exact AdamW remainder",
            &bias.tensor().to_vec()?,
            &expected_bias,
            2e-5,
        );

        let foreign = Parameter::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0],
            [row.of(2), column.of(2)],
            &device,
        )?);
        let mut missing =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&foreign)], 0.02, 0.01, 0.0)?;
        let before_weight = weight.tensor().to_vec()?;
        let before_bias = bias.tensor().to_vec()?;
        let error = missing.step(&mut model).unwrap_err().to_string();
        assert!(error.contains("do not belong to this model"), "{error}");
        assert_eq!(missing.completed_steps(), 0);
        close(
            "missing ID preserves weight",
            &weight.tensor().to_vec()?,
            &before_weight,
            0.0,
        );
        close(
            "missing ID preserves bias",
            &bias.tensor().to_vec()?,
            &before_bias,
            0.0,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn hybrid_rejects_rank_and_auxiliary_failures_without_committing() -> Result<()> {
        let device = Device::cuda(0)?;
        let (row, column, bias_axis) = (Axis::new("row"), Axis::new("column"), Axis::new("bias"));
        let weight = Parameter::new(Tensor::from_slice(
            &[0.2, -0.1, 0.3, 0.5],
            [row.of(2), column.of(2)],
            &device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &[0.4, -0.6],
            [bias_axis.of(2)],
            &device,
        )?);
        let mut model = Parameters(vec![
            ("weight".into(), weight.clone()),
            ("bias".into(), bias.clone()),
        ]);

        let conflict = MuonWithAuxAdamW::new(
            [
                MuonMatrix::fan_in_fan_out(&weight),
                MuonMatrix::fan_out_fan_in(&weight),
            ],
            0.02,
            0.01,
            0.0,
        )
        .err()
        .expect("conflicting orientation must fail")
        .to_string();
        assert!(
            conflict.contains("conflicting matrix orientations"),
            "{conflict}"
        );

        set_gradient(&bias, &[0.25, -0.5])?;
        let before_weight = weight.tensor().to_vec()?;
        let before_bias = bias.tensor().to_vec()?;
        let mut wrong_rank =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&bias)], 0.02, 0.01, 0.0)?;
        let error = wrong_rank.step(&mut model).unwrap_err().to_string();
        assert!(error.contains("must have rank 2"), "{error}");
        assert_eq!(wrong_rank.completed_steps(), 0);
        close(
            "rank failure weight",
            &weight.tensor().to_vec()?,
            &before_weight,
            0.0,
        );
        close(
            "rank failure bias",
            &bias.tensor().to_vec()?,
            &before_bias,
            0.0,
        );

        weight.zero_grad();
        bias.zero_grad();
        set_gradient(&weight, &[0.3, -0.2, 0.1, 0.4])?;
        let mut missing_aux_gradient =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&weight)], 0.02, 0.01, 0.0)?;
        let error = missing_aux_gradient
            .step(&mut model)
            .unwrap_err()
            .to_string();
        assert!(error.contains("parameter has no gradient"), "{error}");
        assert_eq!(missing_aux_gradient.completed_steps(), 0);
        close(
            "auxiliary failure weight",
            &weight.tensor().to_vec()?,
            &before_weight,
            0.0,
        );
        close(
            "auxiliary failure bias",
            &bias.tensor().to_vec()?,
            &before_bias,
            0.0,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn hybrid_failure_after_a_success_is_atomic_against_control() -> Result<()> {
        let device = Device::cuda(0)?;
        let (row, column, bias_axis) = (Axis::new("row"), Axis::new("column"), Axis::new("bias"));
        let make_model = || -> Result<(Parameters, Parameter, Parameter)> {
            let weight = Parameter::new(Tensor::from_slice(
                &[0.2, -0.1, 0.3, 0.5],
                [row.of(2), column.of(2)],
                &device,
            )?);
            let bias = Parameter::new(Tensor::from_slice(
                &[0.4, -0.6],
                [bias_axis.of(2)],
                &device,
            )?);
            Ok((
                Parameters(vec![
                    ("weight".into(), weight.clone()),
                    ("bias".into(), bias.clone()),
                ]),
                weight,
                bias,
            ))
        };
        let (mut subject, subject_weight, subject_bias) = make_model()?;
        let (mut control, control_weight, control_bias) = make_model()?;
        let mut subject_optimizer =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&subject_weight)], 0.02, 0.01, 0.01)?;
        let mut control_optimizer =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&control_weight)], 0.02, 0.01, 0.01)?;

        for (weight, bias) in [
            (&subject_weight, &subject_bias),
            (&control_weight, &control_bias),
        ] {
            set_gradient(weight, &[0.3, -0.2, 0.1, 0.4])?;
            set_gradient(bias, &[0.25, -0.5])?;
        }
        subject_optimizer.step(&mut subject)?;
        control_optimizer.step(&mut control)?;

        subject_weight.zero_grad();
        subject_bias.zero_grad();
        set_gradient(&subject_weight, &[-0.12, 0.31, 0.08, -0.27])?;
        let error = subject_optimizer
            .step(&mut subject)
            .unwrap_err()
            .to_string();
        assert!(error.contains("parameter has no gradient"), "{error}");
        assert_eq!(subject_optimizer.completed_steps(), 1);
        close(
            "post-success failure preserves weight",
            &subject_weight.tensor().to_vec()?,
            &control_weight.tensor().to_vec()?,
            0.0,
        );
        close(
            "post-success failure preserves bias",
            &subject_bias.tensor().to_vec()?,
            &control_bias.tensor().to_vec()?,
            0.0,
        );

        control_weight.zero_grad();
        control_bias.zero_grad();
        set_gradient(&control_weight, &[-0.12, 0.31, 0.08, -0.27])?;
        set_gradient(&control_bias, &[-0.2, 0.15])?;
        set_gradient(&subject_bias, &[-0.2, 0.15])?;
        subject_optimizer.step(&mut subject)?;
        control_optimizer.step(&mut control)?;
        assert_eq!(subject_optimizer.completed_steps(), 2);
        close(
            "recovered weight matches control state",
            &subject_weight.tensor().to_vec()?,
            &control_weight.tensor().to_vec()?,
            3e-5,
        );
        close(
            "recovered bias matches control state",
            &subject_bias.tensor().to_vec()?,
            &control_bias.tensor().to_vec()?,
            3e-5,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn hybrid_allows_an_empty_auxiliary_partition() -> Result<()> {
        let device = Device::cuda(0)?;
        let (row, column) = (Axis::new("row"), Axis::new("column"));
        let weight = Parameter::new(Tensor::from_slice(
            &[0.2, -0.1, 0.3, 0.5],
            [row.of(2), column.of(2)],
            &device,
        )?);
        let mut model = Parameters(vec![("weight".into(), weight.clone())]);
        set_gradient(&weight, &[0.3, -0.2, 0.1, 0.4])?;
        let mut optimizer =
            MuonWithAuxAdamW::new([MuonMatrix::axis_linear(&weight)], 0.02, 0.01, 0.0)?;
        optimizer.step(&mut model)?;
        assert_eq!(optimizer.completed_steps(), 1);
        assert_eq!(optimizer.adamw().completed_steps(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn tiled_l2_normalization_matches_scalar_oracle() -> Result<()> {
        let device = Device::cuda(0)?;
        let axis = Axis::new("value");
        let values = [3.0_f32, -4.0, 12.0];
        let tensor = Tensor::from_slice(&values, [axis.of(values.len())], &device)?;
        let actual = tensor.normalized_l2(1e-7)?.to_vec()?;
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let expected: Vec<_> = values.iter().map(|value| value / (norm + 1e-7)).collect();
        close("tiled L2 normalization", &actual, &expected, 2e-6);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn tiled_l2_normalization_bypasses_generic_plan_cap() -> Result<()> {
        const ROWS: usize = 4_097;
        const COLUMNS: usize = 4_096;
        const { assert!(ROWS * COLUMNS > 16_777_216) };
        let device = Device::cuda(0)?;
        let tensor = Tensor::zeros_for_test(
            [Axis::new("row").of(ROWS), Axis::new("column").of(COLUMNS)],
            &device,
        )?;
        let _normalized = tensor.normalized_l2(1e-7)?;
        device.synchronize()?;
        Ok(())
    }
}
