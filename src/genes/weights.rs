use crate::rng::NeatRng;
use rand::random;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weight(pub f64);

impl Default for Weight {
    fn default() -> Self {
        Weight(random::<f64>() * 2.0 - 1.0)
    }
}

impl Weight {
    pub fn abs(&self) -> f64 {
        self.0.abs()
    }

    pub fn difference(&self, other: &Weight) -> f64 {
        (self.0 - other.0).abs()
    }

    #[inline]
    pub fn perturbate(&mut self, rng: &mut NeatRng) {
        self.0 += rng.weight_perturbation()
    }
}
