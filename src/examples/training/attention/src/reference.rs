//! Independent scalar f64 forward; gradients are checked by central differences.
pub struct Shape {
    pub batch: usize,
    pub time: usize,
    pub heads: usize,
    pub width: usize,
}

impl Shape {
    pub fn attention(&self, q: &[f64], k: &[f64], v: &[f64]) -> Vec<f64> {
        let index = |b, t, h, d| ((b * self.time + t) * self.heads + h) * self.width + d;
        let mut out = vec![0.0; q.len()];
        for b in 0..self.batch {
            for h in 0..self.heads {
                for t in 0..self.time {
                    let scores: Vec<_> = (0..=t)
                        .map(|s| {
                            (0..self.width)
                                .map(|d| q[index(b, t, h, d)] * k[index(b, s, h, d)])
                                .sum::<f64>()
                                / (self.width as f64).sqrt()
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
                    let total: f64 = exps.iter().sum();
                    for d in 0..self.width {
                        out[index(b, t, h, d)] = (0..=t)
                            .map(|s| exps[s] / total * v[index(b, s, h, d)])
                            .sum();
                    }
                }
            }
        }
        out
    }
}
