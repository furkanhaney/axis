use crate::{Axis, Device, Result, Tensor};

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index}: {a:?} != {b:?}");
    }
}

#[test]
#[ignore = "requires CUDA"]
fn translated_windows_match_independent_values_and_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let axes = [Axis::new("batch"), Axis::new("row"), Axis::new("column")];
    let dims = [3, 5, 7];
    let values: Vec<_> = (0..105).map(|i| (i as f32 - 52.0) / 8.0).collect();
    for reversed in [false, true] {
        for selected in 0..3 {
            // Exercise leading/trailing padding, wider-than-input padding, and crops.
            for (before, after, start, length) in [
                (2, 4, 0, 0),
                (0, 9, 0, 0),
                (8, 0, 0, 0),
                (0, 0, 1, dims[selected] - 1),
                (0, 0, 0, 1),
            ] {
                let input = Tensor::from_slice(
                    &values,
                    axes.iter().zip(dims).map(|(&a, n)| a.of(n)),
                    &device,
                )?;
                let input = if reversed {
                    input.with_layout([axes[2], axes[0], axes[1]])?
                } else {
                    input
                };
                let input = input.with_grad();
                let output = if length == 0 {
                    input.pad_zeros(axes[selected], before, after)?
                } else {
                    input.narrow(axes[selected], start, length)?
                };
                let mut shape = dims;
                shape[selected] = if length == 0 {
                    dims[selected] + before + after
                } else {
                    length
                };
                assert_eq!(output.shape().axes(), axes);
                let count: usize = shape.iter().product();
                let mut expected = vec![0.0; count];
                let mut derivative = vec![0.0_f64; values.len()];
                let weights: Vec<_> = (0..count).map(|i| (i % 19) as f32 - 9.0).collect();
                for b in 0..shape[0] {
                    for y in 0..shape[1] {
                        for x in 0..shape[2] {
                            let out = (b * shape[1] + y) * shape[2] + x;
                            let mut coords = [b as isize, y as isize, x as isize];
                            coords[selected] += if length == 0 {
                                -(before as isize)
                            } else {
                                start as isize
                            };
                            if coords[selected] >= 0 && coords[selected] < dims[selected] as isize {
                                let source = (coords[0] as usize * dims[1] + coords[1] as usize)
                                    * dims[2]
                                    + coords[2] as usize;
                                expected[out] = values[source];
                                derivative[source] = f64::from(weights[out]) / count as f64;
                            }
                        }
                    }
                }
                assert_bits(&output.to_vec()?, &expected);
                let upstream = Tensor::from_slice(
                    &weights,
                    axes.iter().zip(shape).map(|(&a, n)| a.of(n)),
                    &device,
                )?;
                output.mul(&upstream)?.mean(axes)?.backward()?;
                for (&actual, &expected) in input.grad().unwrap().to_vec()?.iter().zip(&derivative)
                {
                    assert!((f64::from(actual) - expected).abs() < 2e-7);
                }
            }
        }
    }
    println!("window values and gradients PASS: 30 cases, 3 selected axes, 2 layouts");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn translated_windows_preserve_special_values_and_reject_invalid_geometry() -> Result<()> {
    let device = Device::cuda(0)?;
    let axis = Axis::new("value");
    let missing = Axis::new("value");
    let values = [
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc12345),
        7.0,
    ];
    let input = Tensor::from_slice(&values, [axis.of(5)], &device)?;
    let padded = input.pad_zeros(axis, 7, 9)?;
    let mut expected = vec![0.0; 21];
    expected[7..12].copy_from_slice(&values);
    assert_bits(&padded.to_vec()?, &expected);
    assert_bits(&padded.narrow(axis, 7, 5)?.to_vec()?, &values);
    assert_bits(
        &input.pad_zeros(axis, 0, 0)?.narrow(axis, 0, 5)?.to_vec()?,
        &values,
    );
    assert!(input.pad_zeros(missing, 0, 0).is_err());
    assert!(input.narrow(missing, 0, 1).is_err());
    assert!(input.narrow(axis, 0, 0).is_err());
    assert!(input.narrow(axis, 5, 1).is_err());
    assert!(input.narrow(axis, usize::MAX, 2).is_err());
    assert!(input.pad_zeros(axis, usize::MAX, 1).is_err());
    assert!(input.pad_zeros(axis, 0, i32::MAX as usize).is_err());
    assert!(input.pad_zeros(axis, i32::MAX as usize, 0).is_err());
    // Masked-away nonfinite upstream derivatives must become exact zeros.
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0], [axis.of(3)], &device)?.with_grad();
    let g = Tensor::from_slice(
        &[f32::NAN, 6.0, 9.0, 12.0, f32::INFINITY],
        [axis.of(5)],
        &device,
    )?;
    x.pad_zeros(axis, 1, 1)?.mul(&g)?.mean(axis)?.backward()?;
    for (a, b) in x.grad().unwrap().to_vec()?.iter().zip([1.2, 1.8, 2.4]) {
        assert!((a - b).abs() < 5e-7);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn translated_windows_cover_tinyvit_geometry_and_exceed_generic_plan_ceiling() -> Result<()> {
    let device = Device::cuda(0)?;
    let (y, x, c) = (Axis::new("y"), Axis::new("x"), Axis::new("channel"));
    for (side, channels, window) in [(128usize, 128, 7), (64, 160, 14), (64, 320, 7)] {
        let values: Vec<_> = (0..side * side * channels)
            .map(|i| (i % 251) as f32 - 125.0)
            .collect();
        let input = Tensor::from_slice(&values, [y.of(side), x.of(side), c.of(channels)], &device)?;
        let pad = (window - side % window) % window;
        let output = input.pad_zeros(y, 0, pad)?.pad_zeros(x, 0, pad)?;
        let actual = output.to_vec()?;
        for row in 0..side + pad {
            for column in 0..side + pad {
                let begin = (row * (side + pad) + column) * channels;
                let expected = if row < side && column < side {
                    &values[(row * side + column) * channels..(row * side + column + 1) * channels]
                } else {
                    &vec![0.0; channels]
                };
                assert_bits(&actual[begin..begin + channels], expected);
            }
        }
        assert_bits(
            &output.narrow(y, 0, side)?.narrow(x, 0, side)?.to_vec()?,
            &values,
        );
        println!("TinyViT window PASS {side}x{side}x{channels}, bottom/right padding={pad}");
    }
    // The compact path must not inherit the 16,777,216-entry generic plan ceiling.
    let large = Tensor::zeros([x.of(16_777_217)], &device)?;
    let padded = large.pad_zeros(x, 1, 3)?;
    assert_eq!(padded.extent(x)?, 16_777_221);
    assert_bits(&padded.narrow(x, 16_777_216, 5)?.to_vec()?, &[0.0; 5]);
    println!("window compact geometry PASS beyond generic plan ceiling");
    Ok(())
}
