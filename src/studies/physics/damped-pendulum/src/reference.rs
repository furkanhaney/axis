//! Independent f64 damped-pendulum oracle.

#[derive(Clone, Copy, Debug)]
pub struct Pendulum {
    pub gravity: f64,
    pub length: f64,
    pub damping_rate: f64,
    pub initial_angle: f64,
    pub initial_angular_velocity: f64,
}

impl Pendulum {
    pub fn acceleration(self, angle: f64, angular_velocity: f64) -> f64 {
        -self.damping_rate * angular_velocity - (self.gravity / self.length) * angle.sin()
    }

    fn derivative(self, state: [f64; 2]) -> [f64; 2] {
        [state[1], self.acceleration(state[0], state[1])]
    }

    fn rk4_step(self, state: [f64; 2], step: f64) -> [f64; 2] {
        let k1 = self.derivative(state);
        let k2 = self.derivative([state[0] + 0.5 * step * k1[0], state[1] + 0.5 * step * k1[1]]);
        let k3 = self.derivative([state[0] + 0.5 * step * k2[0], state[1] + 0.5 * step * k2[1]]);
        let k4 = self.derivative([state[0] + step * k3[0], state[1] + step * k3[1]]);
        [
            state[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        ]
    }

    /// Integrate independently to `time`, using steps no larger than `max_step`.
    pub fn state_at(self, time: f64, max_step: f64) -> [f64; 2] {
        assert!(time.is_finite() && time >= 0.0);
        assert!(max_step.is_finite() && max_step > 0.0);
        let steps = ((time / max_step).ceil() as usize).max(1);
        let step = time / steps as f64;
        let mut state = [self.initial_angle, self.initial_angular_velocity];
        for _ in 0..steps {
            state = self.rk4_step(state, step);
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_converges_and_preserves_the_initial_condition() {
        let system = Pendulum {
            gravity: 9.81,
            length: 1.0,
            damping_rate: 0.2,
            initial_angle: 0.7,
            initial_angular_velocity: -0.3,
        };
        assert_eq!(system.state_at(0.0, 1e-3), [0.7, -0.3]);
        let coarse = system.state_at(2.0, 2e-3);
        let fine = system.state_at(2.0, 1e-3);
        assert!((coarse[0] - fine[0]).abs() < 1e-10);
        assert!((coarse[1] - fine[1]).abs() < 1e-9);
    }

    #[test]
    fn normalized_damping_equation_has_no_absolute_mass() {
        let system = Pendulum {
            gravity: 9.81,
            length: 0.8,
            damping_rate: 0.35,
            initial_angle: 0.5,
            initial_angular_velocity: 0.2,
        };
        // Caliper used b = gamma * m * L^2. Dividing the torque equation by
        // m * L^2 leaves gamma, so absolute mass cannot affect this oracle.
        let expected = -0.35 * 0.2 - (9.81 / 0.8) * 0.5_f64.sin();
        assert!((system.acceleration(0.5, 0.2) - expected).abs() < 1e-14);
    }
}
