//! Explicit forward-kernel specializations prepared before a latency-sensitive step.
use crate::Result;

/// Physical kernel shapes for [`crate::Device::prepare_kernels`].
///
/// These describe contiguous lowered operations, not named tensor dimensions.
/// Preparation currently covers forward matrix products and softmax. Layout
/// copies, other operators, and backward kernels remain lazily compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelSpec {
    /// Batched `[batch, rows, inner]` by `[batch, inner, columns]` product.
    Matmul {
        batch: usize,
        rows: usize,
        inner: usize,
        columns: usize,
    },
    /// Independent contiguous softmax rows.
    Softmax { rows: usize, width: usize },
}

impl KernelSpec {
    pub(crate) fn validate(self) -> Result<()> {
        let check = |dims: &[usize]| -> Result<()> {
            let count = dims
                .iter()
                .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
                .ok_or("kernel preparation requires positive, non-overflowing dimensions")?;
            if count > i32::MAX as usize {
                return Err("kernel preparation exceeds signed 32-bit indexing".into());
            }
            Ok(())
        };
        match self {
            Self::Matmul {
                batch,
                rows,
                inner,
                columns,
            } => {
                check(&[batch, rows, inner])?;
                check(&[batch, inner, columns])?;
                check(&[batch, rows, columns])
            }
            Self::Softmax { rows, width } => {
                check(&[rows, width])?;
                if width.next_power_of_two() > i32::MAX as usize {
                    return Err(
                        "kernel preparation softmax tile exceeds signed 32-bit indexing".into(),
                    );
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preparation_rejects_invalid_shapes_before_cuda() {
        assert!(
            KernelSpec::Matmul {
                batch: 0,
                rows: 1,
                inner: 1,
                columns: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            KernelSpec::Matmul {
                batch: 1,
                rows: usize::MAX,
                inner: 2,
                columns: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            KernelSpec::Softmax { rows: 1, width: 0 }
                .validate()
                .is_err()
        );
        assert!(
            KernelSpec::Softmax {
                rows: 1,
                width: i32::MAX as usize
            }
            .validate()
            .is_err()
        );
        assert!(
            KernelSpec::Matmul {
                batch: 2,
                rows: 3,
                inner: 5,
                columns: 7
            }
            .validate()
            .is_ok()
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::{Axis, Device, KernelSpec, Result, Tensor, jit_cache_stats};
    use std::{process::Command, time::Instant};

    #[test]
    #[ignore = "requires CUDA"]
    fn preparation_validates_all_shapes_and_covers_bf16() -> Result<()> {
        let device = Device::cuda_bf16(0)?;
        let spec = KernelSpec::Matmul {
            batch: 1,
            rows: 2,
            inner: 3,
            columns: 2,
        };
        let before = jit_cache_stats();
        assert!(
            device
                .prepare_kernels(&[spec, KernelSpec::Softmax { rows: 0, width: 3 }])
                .is_err()
        );
        assert_eq!(jit_cache_stats(), before);
        device.prepare_kernels(&[])?;
        device.prepare_kernels(&[spec])?;
        let prepared = jit_cache_stats();
        let (r, k, c) = (Axis::new("row"), Axis::new("inner"), Axis::new("column"));
        let left = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], [r.of(2), k.of(3)], &device)?;
        let right = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], [k.of(3), c.of(2)], &device)?;
        assert_eq!(left.contract(&right, k)?.to_vec()?, [22., 28., 49., 64.]);
        assert_eq!(jit_cache_stats(), prepared);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn prepared_kernels_match_lazy_cold_and_warm_execution() -> Result<()> {
        if let Ok(mode) = std::env::var("AXIS_PREPARATION_TEST_MODE") {
            let device = Device::cuda(0)?;
            let shapes: Vec<_> = (0..30).map(|i| (3, 17 + i, 9 + i)).collect();
            let specs: Vec<_> = shapes
                .iter()
                .flat_map(|&(rows, inner, columns)| {
                    [
                        KernelSpec::Matmul {
                            batch: 1,
                            rows,
                            inner,
                            columns,
                        },
                        KernelSpec::Softmax {
                            rows,
                            width: columns,
                        },
                    ]
                })
                .collect();
            let prepare_start = Instant::now();
            if mode.starts_with("prepared") {
                device.prepare_kernels(&specs)?;
            }
            let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
            let before = jit_cache_stats();
            let started = Instant::now();
            let mut fingerprint = 0u64;
            for (rows, inner, columns) in shapes {
                let (r, k, c) = (Axis::new("row"), Axis::new("inner"), Axis::new("column"));
                let lv: Vec<_> = (0..rows * inner)
                    .map(|i| ((i % 7) as f32 - 3.0) / 16.0)
                    .collect();
                let rv: Vec<_> = (0..inner * columns)
                    .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
                    .collect();
                let left = Tensor::from_slice(&lv, [r.of(rows), k.of(inner)], &device)?;
                let right = Tensor::from_slice(&rv, [k.of(inner), c.of(columns)], &device)?;
                let values = left.contract(&right, k)?.softmax(c)?.to_vec()?;
                for row in 0..rows {
                    let logits: Vec<f64> = (0..columns)
                        .map(|col| {
                            (0..inner)
                                .map(|j| {
                                    f64::from(lv[row * inner + j])
                                        * f64::from(rv[j * columns + col])
                                })
                                .sum()
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
                    for col in 0..columns {
                        let actual = values[row * columns + col];
                        assert!((f64::from(actual) - (logits[col] - max).exp() / sum).abs() < 2e-6);
                        fingerprint = fingerprint.rotate_left(7) ^ u64::from(actual.to_bits());
                    }
                }
            }
            let first_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after = jit_cache_stats();
            if mode.starts_with("prepared") {
                assert_eq!(
                    after, before,
                    "prepared shapes must not invoke the disk/compile path at execution"
                );
            }
            if mode.ends_with("warm") {
                assert_eq!(after.misses, 0, "warm process must reuse the disk cache");
            }
            println!(
                "PREPARATION mode={mode} prepare_ms={prepare_ms:.3} first_step_ms={first_ms:.3} hits={} misses={} fingerprint={fingerprint}",
                after.hits, after.misses
            );
            return Ok(());
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build")
            .join(format!("preparation-test-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let mut fingerprints = Vec::new();
        for mode in ["lazy-cold", "prepared-cold", "lazy-warm", "prepared-warm"] {
            let output = Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "preparation::cuda_tests::prepared_kernels_match_lazy_cold_and_warm_execution",
                    "--ignored",
                    "--nocapture",
                ])
                .env("AXIS_PREPARATION_TEST_MODE", mode)
                .env("AXIS_JIT_CACHE", "on")
                .env(
                    "AXIS_JIT_CACHE_DIR",
                    root.join(if mode.starts_with("lazy") {
                        "lazy"
                    } else {
                        "prepared"
                    }),
                )
                .output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let receipt = stdout
                .lines()
                .find(|line| line.starts_with("PREPARATION "))
                .expect("child receipt");
            println!("{receipt}");
            fingerprints.push(receipt.split("fingerprint=").nth(1).unwrap().to_owned());
        }
        assert!(fingerprints.iter().all(|value| value == &fingerprints[0]));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
