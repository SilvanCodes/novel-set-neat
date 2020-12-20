use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weight(pub f64);

impl Deref for Weight {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Weight {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
