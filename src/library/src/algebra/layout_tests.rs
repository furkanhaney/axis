use crate::{Axis, Device, Result, Shape, Tensor};

const ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[test]
#[ignore = "requires CUDA"]
fn compact_layout_identity_transforms_are_storage_views() -> Result<()> {
    let device = Device::cuda(0)?;
    let (a, b, c, merged, p, q) = (
        Axis::new("a"),
        Axis::new("b"),
        Axis::new("c"),
        Axis::new("merged"),
        Axis::new("p"),
        Axis::new("q"),
    );
    let input = Tensor::zeros([a.of(2), b.of(3), c.of(5)], &device)?;
    assert!(input.with_layout([a, b, c])?.shares_buffer(&input));
    assert!(input.merge([a, b], merged)?.shares_buffer(&input));
    assert!(input.split(b, [p.of(1), q.of(3)])?.shares_buffer(&input));
    assert!(!input.with_layout([c, b, a])?.shares_buffer(&input));
    let singleton = Tensor::from_slice(
        &[0., 1., 2., 3., 4., 5.],
        [a.of(2), b.of(1), c.of(3)],
        &device,
    )?;
    let reordered = singleton.with_layout([b, a, c])?;
    assert!(reordered.shares_buffer(&singleton));
    assert_eq!(reordered.layout_strides(), &[3, 6, 1]);
    let merged = singleton.merge([b, a], merged)?;
    assert!(merged.shares_buffer(&singleton));
    assert_eq!(merged.to_vec()?, [0., 1., 2., 3., 4., 5.]);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn compact_layout_permutations_and_merges_preserve_values_and_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let axes = [Axis::new("a"), Axis::new("b"), Axis::new("c")];
    let dims = [2, 3, 5];
    let values: Vec<_> = (0..30).map(|i| i as f32 - 11.0).collect();
    let weights: Vec<_> = (0..30).map(|i| (i + 1) as f32).collect();
    let merged = Axis::new("merged");
    let mut cases = 0;
    for from in ORDERS {
        for to in ORDERS {
            let input = Tensor::from_slice(&values, (0..3).map(|i| axes[i].of(dims[i])), &device)?
                .with_layout(from.map(|i| axes[i]))?
                .with_grad();
            let output = input.with_layout(to.map(|i| axes[i]))?;
            assert_eq!(output.shape(), input.shape());
            assert_eq!(output.to_vec()?, values);
            let weight =
                Tensor::from_slice(&weights, input.shape().dims().iter().copied(), &device)?;
            output.mul(&weight)?.mean(axes)?.backward()?;
            for (actual, expected) in input.grad().unwrap().to_vec()?.iter().zip(&weights) {
                assert!((actual - expected / 30.0).abs() < 2e-6);
            }
            cases += 1;
        }
        // Every ordered pair includes both adjacent and nonadjacent merges.
        for selected in ORDERS.map(|p| [p[0], p[1]]) {
            let remaining = (0..3).find(|i| !selected.contains(i)).unwrap();
            let mut out_dims = Vec::new();
            for i in 0..3 {
                if i == *selected.iter().min().unwrap() {
                    out_dims.push(merged.of(dims[selected[0]] * dims[selected[1]]));
                } else if i == remaining {
                    out_dims.push(axes[i].of(dims[i]));
                }
            }
            let shape = Shape::new(out_dims)?;
            let input = Tensor::from_slice(&values, (0..3).map(|i| axes[i].of(dims[i])), &device)?
                .with_layout(from.map(|i| axes[i]))?
                .with_grad();
            let output = input.merge(selected.map(|i| axes[i]), merged)?;
            assert_eq!(output.shape(), &shape);
            let mut expected = vec![0.0; 30];
            let mut gradient = vec![0.0; 30];
            for a in 0..2 {
                for b in 0..3 {
                    for c in 0..5 {
                        let coord = [a, b, c];
                        let input_index = (a * 3 + b) * 5 + c;
                        let combined = coord[selected[0]] * dims[selected[1]] + coord[selected[1]];
                        let output_index = if shape.dims()[0].axis == merged {
                            combined * dims[remaining] + coord[remaining]
                        } else {
                            coord[remaining] * dims[selected[0]] * dims[selected[1]] + combined
                        };
                        expected[output_index] = values[input_index];
                        gradient[input_index] = weights[output_index] / 30.0;
                    }
                }
            }
            assert_eq!(output.to_vec()?, expected);
            let weight = Tensor::from_slice(&weights, shape.dims().iter().copied(), &device)?;
            output.mul(&weight)?.mean(shape.axes())?.backward()?;
            for (&actual, &expected) in input.grad().unwrap().to_vec()?.iter().zip(&gradient) {
                assert!((actual - expected).abs() < 2e-6);
            }
            cases += 1;
        }
    }
    assert!(Tensor::layout_metadata_max() <= 9);
    assert_eq!(cases, 72);
    println!("compact layout PASS: {cases} independently indexed value/gradient cases");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn compact_layout_split_and_special_values() -> Result<()> {
    let device = Device::cuda(0)?;
    let (a, b, c, p, q) = (
        Axis::new("a"),
        Axis::new("b"),
        Axis::new("c"),
        Axis::new("p"),
        Axis::new("q"),
    );
    let bits = [
        0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000,
    ];
    let values: Vec<_> = (0..30)
        .map(|i| f32::from_bits(bits[i % bits.len()]))
        .collect();
    let input = Tensor::from_slice(&values, [a.of(2), b.of(3), c.of(5)], &device)?;
    let permuted = input.with_layout([c, a, b])?;
    assert_eq!(
        permuted
            .to_vec()?
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let input = Tensor::from_slice(
        &(0..30).map(|i| i as f32).collect::<Vec<_>>(),
        [a.of(2), b.of(15)],
        &device,
    )?
    .with_layout([b, a])?
    .with_grad();
    let split = input.split(b, [p.of(3), q.of(5)])?;
    assert_eq!(split.shape(), &Shape::new([a.of(2), p.of(3), q.of(5)])?);
    assert_eq!(
        split.to_vec()?,
        (0..30).map(|i| i as f32).collect::<Vec<_>>()
    );
    split
        .squared_error(&Tensor::zeros(
            split.shape().dims().iter().copied(),
            &device,
        )?)?
        .mean([a, p, q])?
        .backward()?;
    for (i, &actual) in input.grad().unwrap().to_vec()?.iter().enumerate() {
        assert!((actual - 2.0 * i as f32 / 30.0).abs() < 2e-6);
    }
    assert!(input.with_layout([a]).is_err());
    assert!(input.with_layout([a, a]).is_err());
    assert!(input.merge([a, a], p).is_err());
    assert!(input.merge([], p).is_err());
    assert!(input.merge(a, b).is_err());
    assert!(input.split(b, [p.of(4), q.of(4)]).is_err());
    assert!(input.split(b, [a.of(3), q.of(5)]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn compact_layout_exceeds_generic_plan_ceiling_with_rank_sized_metadata() -> Result<()> {
    let device = Device::cuda(0)?;
    let (a, b, c, p, q, merged) = (
        Axis::new("a"),
        Axis::new("b"),
        Axis::new("c"),
        Axis::new("p"),
        Axis::new("q"),
        Axis::new("merged"),
    );
    let count = 2 * 2049 * 4097;
    assert!(count > 16_777_216);
    let input = Tensor::from_slice(
        &(0..count).map(|i| (i % 101) as f32).collect::<Vec<_>>(),
        [a.of(2), b.of(2049), c.of(4097)],
        &device,
    )?
    .with_grad();
    let reordered = input.with_layout([c, a, b])?;
    let split = reordered.split(c, [p.of(17), q.of(241)])?;
    let combined = reordered.merge([c, a], merged)?;
    for [ai, bi, ci] in [[0, 0, 0], [1, 2048, 4096], [1, 113, 71], [0, 1741, 3317]] {
        let expected = (((ai * 2049 + bi) * 4097 + ci) % 101) as f32;
        assert_eq!(
            reordered
                .select(a, ai)?
                .select(b, bi)?
                .select(c, ci)?
                .item()?,
            expected
        );
        assert_eq!(
            split
                .select(a, ai)?
                .select(b, bi)?
                .select(p, ci / 241)?
                .select(q, ci % 241)?
                .item()?,
            expected
        );
        assert_eq!(
            combined
                .select(merged, ci * 2 + ai)?
                .select(b, bi)?
                .item()?,
            expected
        );
    }
    combined
        .select(merged, 71 * 2 + 1)?
        .select(b, 113)?
        .backward()?;
    let gradient = input.grad().ok_or("missing large-layout gradient")?;
    assert_eq!(
        gradient
            .select(a, 1)?
            .select(b, 113)?
            .select(c, 71)?
            .item()?,
        1.0
    );
    assert_eq!(
        gradient
            .select(a, 0)?
            .select(b, 113)?
            .select(c, 71)?
            .item()?,
        0.0
    );
    assert_eq!(
        gradient
            .select(a, 1)?
            .select(b, 112)?
            .select(c, 71)?
            .item()?,
        0.0
    );
    assert!(Tensor::layout_metadata_max() <= 9);
    println!("compact layout PASS: {count} values, <=9 host metadata integers per permutation");
    Ok(())
}
