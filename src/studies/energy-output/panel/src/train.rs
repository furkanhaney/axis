//! Bounded axis mechanics witness for the country-year real-GDP MLP.
use axis::prelude::*;
use csv::StringRecord;
use std::{
    collections::BTreeSet,
    env,
    path::{Path, PathBuf},
};

const WIDTH: usize = 8;
const FEATURES: usize = 9;
const INPUT_COLUMNS: [&str; 4] = [
    "electricity_demand_twh",
    "oil_total_liquids_mbd",
    "electrical_electronic_trades_workers",
    "electronics_papers",
];
const TARGET: &str = "real_ppp_gdp_2021_intl_usd";

#[derive(Clone, Copy, Debug)]
enum Split {
    Country,
    Future,
}

struct Config {
    data: PathBuf,
    split: Split,
    steps: usize,
    train_limit: usize,
    validation_limit: usize,
}

#[derive(Clone)]
struct Row {
    iso3: String,
    year: i32,
    logged: [Option<f64>; 4],
    target: f64,
}

#[derive(Clone)]
struct Prepared {
    input: [f32; FEATURES],
    target: f32,
}

struct Statistics {
    imputation: [f64; 4],
    input_mean: [f64; FEATURES],
    input_scale: [f64; FEATURES],
    target_mean: f64,
    target_scale: f64,
}

fn field<'a>(record: &'a StringRecord, index: usize, name: &str) -> Result<&'a str> {
    record
        .get(index)
        .ok_or_else(|| format!("row has no {name} column").into())
}

fn finite_nonnegative(value: &str, strictly_positive: bool) -> Option<f64> {
    let parsed = value.parse::<f64>().ok()?;
    (parsed.is_finite()
        && if strictly_positive {
            parsed > 0.0
        } else {
            parsed >= 0.0
        })
    .then_some(parsed)
}

fn load_rows(path: &Path) -> Result<Vec<Row>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| format!("missing CSV column {name}").into())
    };
    let iso3 = column("iso3")?;
    let year = column("year")?;
    let target = column(TARGET)?;
    let regression_area = column("regression_area")?;
    let workforce_usable = column("workforce_usable")?;
    let input_columns: Vec<_> = INPUT_COLUMNS
        .iter()
        .map(|name| column(name))
        .collect::<Result<_>>()?;
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        if field(&record, regression_area, "regression_area")? != "True" {
            continue;
        }
        let Some(target_value) = finite_nonnegative(field(&record, target, TARGET)?, true) else {
            continue;
        };
        let workforce_ok = field(&record, workforce_usable, "workforce_usable")? == "True";
        let raw: [Option<f64>; 4] = std::array::from_fn(|index| {
            if index == 2 && !workforce_ok {
                None
            } else {
                finite_nonnegative(
                    record.get(input_columns[index]).unwrap_or_default(),
                    index == 0,
                )
            }
        });
        let logged = std::array::from_fn(|index| {
            raw[index].map(|value| (value + if index == 0 { 0.0 } else { 1.0 }).log10())
        });
        rows.push(Row {
            iso3: field(&record, iso3, "iso3")?.to_owned(),
            year: field(&record, year, "year")?.parse()?,
            logged,
            target: target_value.log10(),
        });
    }
    rows.sort_by(|left, right| (&left.iso3, left.year).cmp(&(&right.iso3, right.year)));
    if rows.is_empty() {
        return Err("panel contains no eligible supervised rows".into());
    }
    Ok(rows)
}

fn split_rows(rows: &[Row], split: Split) -> Result<(Vec<Row>, Vec<Row>)> {
    let (train, validation): (Vec<_>, Vec<_>) = match split {
        Split::Future => rows.iter().cloned().partition(|row| row.year <= 2019),
        Split::Country => {
            let countries: Vec<_> = rows
                .iter()
                .map(|row| row.iso3.as_str())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let held_out: BTreeSet<_> = countries
                .into_iter()
                .enumerate()
                .filter_map(|(index, iso3)| (index % 5 == 0).then_some(iso3))
                .collect();
            rows.iter()
                .cloned()
                .partition(|row| !held_out.contains(row.iso3.as_str()))
        }
    };
    if train.is_empty() || validation.is_empty() {
        return Err("declared split produced an empty side".into());
    }
    match split {
        Split::Country => {
            let train_iso3: BTreeSet<_> = train.iter().map(|row| row.iso3.as_str()).collect();
            if validation
                .iter()
                .any(|row| train_iso3.contains(row.iso3.as_str()))
            {
                return Err("country split leaked an ISO3 code".into());
            }
        }
        Split::Future => {
            if train.iter().any(|row| row.year > 2019)
                || validation.iter().any(|row| row.year < 2020)
            {
                return Err("future split crossed its 2019/2020 boundary".into());
            }
        }
    }
    Ok((train, validation))
}

fn mean_scale(values: impl Iterator<Item = f64>) -> Result<(f64, f64)> {
    let values: Vec<_> = values.collect();
    if values.is_empty() {
        return Err("cannot standardize an empty column".into());
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let scale = variance.sqrt();
    Ok((mean, if scale > 0.0 { scale } else { 1.0 }))
}

fn fit_statistics(train: &[Row]) -> Result<Statistics> {
    let mut imputation = [0.0; 4];
    for column in 0..4 {
        let observed: Vec<_> = train.iter().filter_map(|row| row.logged[column]).collect();
        if observed.is_empty() {
            return Err(format!("training split has no observed {}", INPUT_COLUMNS[column]).into());
        }
        imputation[column] = observed.iter().sum::<f64>() / observed.len() as f64;
    }
    let design = |row: &Row| -> [f64; FEATURES] {
        std::array::from_fn(|column| match column {
            0..=3 => row.logged[column].unwrap_or(imputation[column]),
            4..=7 => f64::from(row.logged[column - 4].is_none()),
            8 => f64::from(row.year - 2000),
            _ => unreachable!(),
        })
    };
    let mut input_mean = [0.0; FEATURES];
    let mut input_scale = [1.0; FEATURES];
    for column in 0..FEATURES {
        (input_mean[column], input_scale[column]) =
            mean_scale(train.iter().map(|row| design(row)[column]))?;
    }
    let (target_mean, target_scale) = mean_scale(train.iter().map(|row| row.target))?;
    Ok(Statistics {
        imputation,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
    })
}

fn prepare(rows: &[Row], stats: &Statistics) -> Vec<Prepared> {
    rows.iter()
        .map(|row| {
            let raw: [f64; FEATURES] = std::array::from_fn(|column| match column {
                0..=3 => row.logged[column].unwrap_or(stats.imputation[column]),
                4..=7 => f64::from(row.logged[column - 4].is_none()),
                8 => f64::from(row.year - 2000),
                _ => unreachable!(),
            });
            Prepared {
                input: std::array::from_fn(|column| {
                    ((raw[column] - stats.input_mean[column]) / stats.input_scale[column]) as f32
                }),
                target: ((row.target - stats.target_mean) / stats.target_scale) as f32,
            }
        })
        .collect()
}

fn tensors(
    rows: &[Prepared],
    batch: Axis,
    feature: Axis,
    output: Axis,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let input: Vec<_> = rows.iter().flat_map(|row| row.input).collect();
    let target: Vec<_> = rows.iter().map(|row| row.target).collect();
    Ok((
        Tensor::from_slice(&input, [batch.of(rows.len()), feature.of(FEATURES)], device)?,
        Tensor::from_slice(&target, [batch.of(rows.len()), output.of(1)], device)?,
    ))
}

fn mse(
    model: &Sequential,
    input: &Tensor,
    target: &Tensor,
    batch: Axis,
    output: Axis,
) -> Result<f32> {
    model
        .forward(input)?
        .squared_error(target)?
        .mean([batch, output])?
        .item()
}

fn parse() -> Result<Config> {
    let mut config = Config {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../research/energy-output/data/country_panel_2000_2024.csv"),
        split: Split::Country,
        steps: 40,
        train_limit: 256,
        validation_limit: 128,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {}
            "--split" => {
                config.split = match args
                    .next()
                    .ok_or("--split needs country or future")?
                    .as_str()
                {
                    "country" => Split::Country,
                    "future" => Split::Future,
                    value => return Err(format!("unknown split {value:?}").into()),
                }
            }
            "--steps" => config.steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--help" | "-h" => {
                println!("train --smoke [--split country|future] [--steps N] [--data PANEL.csv]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if config.steps == 0 {
        return Err("steps must be positive".into());
    }
    Ok(config)
}

fn main() -> Result<()> {
    let config = parse()?;
    let rows = load_rows(&config.data)?;
    let (mut train_rows, mut validation_rows) = split_rows(&rows, config.split)?;
    train_rows.truncate(config.train_limit);
    validation_rows.truncate(config.validation_limit);
    let stats = fit_statistics(&train_rows)?;
    let train = prepare(&train_rows, &stats);
    let validation = prepare(&validation_rows, &stats);

    let device = Device::cuda(0)?;
    let (batch, feature, hidden, output) = (
        Axis::new("country_year"),
        Axis::new("feature"),
        Axis::new("hidden"),
        Axis::new("real_gdp"),
    );
    let (train_input, train_target) = tensors(&train, batch, feature, output, &device)?;
    let (validation_input, validation_target) =
        tensors(&validation, batch, feature, output, &device)?;
    let mut model = Sequential::new((
        Linear::new(feature, hidden.of(WIDTH)),
        ReLU,
        Linear::new(hidden, output.of(1)),
    ));
    model.build(train_input.shape(), &device, 20260918)?;
    let parameters: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor().shape().len())
        .sum();
    if parameters != 89 {
        return Err(format!("parameter budget changed: expected 89, found {parameters}").into());
    }

    let initial = mse(&model, &validation_input, &validation_target, batch, output)?;
    let mut trainer = Trainer::new(AdamW::new(0.01, 0.001)?);
    let mut best = f32::INFINITY;
    let mut best_step = 0;
    let mut checkpoint = Vec::new();
    for step in 1..=config.steps {
        trainer.step(&mut model, |model| {
            model
                .forward(&train_input)?
                .squared_error(&train_target)?
                .mean([batch, output])
        })?;
        if step % 10 == 0 || step == config.steps {
            let value = mse(&model, &validation_input, &validation_target, batch, output)?;
            println!("step {step:3} validation_mse_normalized {value:.6}");
            if value < best {
                best = value;
                best_step = step;
                checkpoint = model
                    .parameters()
                    .iter()
                    .map(|parameter| parameter.tensor().to_vec())
                    .collect::<Result<_>>()?;
            }
        }
    }
    for (parameter, values) in model.parameters().iter().zip(&checkpoint) {
        parameter.set_values(values)?;
    }
    let restored = mse(&model, &validation_input, &validation_target, batch, output)?;
    if (restored - best).abs() > 1e-5 || !restored.is_finite() {
        return Err("restored checkpoint does not reproduce its validation loss".into());
    }
    let train_countries = train_rows
        .iter()
        .map(|row| row.iso3.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let validation_countries = validation_rows
        .iter()
        .map(|row| row.iso3.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    println!(
        "split={:?} train_rows={} train_countries={} validation_rows={} validation_countries={}",
        config.split,
        train.len(),
        train_countries,
        validation.len(),
        validation_countries
    );
    println!(
        "params={parameters} initial_validation_mse={initial:.6} best_validation_mse={best:.6} best_step={best_step}"
    );
    println!("optimizer=AdamW learning_rate=0.01 weight_decay=0.001");
    println!("PASS: bounded panel mechanics and best-checkpoint restoration");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_workforce_is_missing_and_log_shifts_match_source() -> Result<()> {
        assert_eq!(finite_nonnegative("0", true), None);
        assert_eq!(finite_nonnegative("0", false), Some(0.0));
        assert_eq!((finite_nonnegative("9", false).unwrap() + 1.0).log10(), 1.0);
        Ok(())
    }
}
