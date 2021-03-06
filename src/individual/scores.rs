use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub trait Score
where
    Self: Sized,
{
    type Value;
    fn value(&self) -> Self::Value;

    fn shift(self, baseline: Self::Value) -> Self;
    fn normalize_with(self, with: Self::Value) -> Self;
}

pub trait ScoreType {}

pub trait ScoreValue {
    type Value;
    fn value(&self) -> Self::Value;
}

#[derive(Debug, Default, Copy, Clone, Deserialize, Serialize, PartialEq)]
pub struct Fitness(pub f64);

impl ScoreValue for Fitness {
    type Value = f64;

    fn value(&self) -> Self::Value {
        self.0
    }
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct FitnessScore {
    pub raw: Raw<Fitness>,
    pub shifted: Shifted<Fitness>,
    pub normalized: Normalized<Fitness>,
}

impl FitnessScore {
    pub fn new(raw: f64, baseline: f64, with: f64) -> Self {
        let raw = Raw::fitness(raw);
        let shifted = raw.shift(baseline);
        let normalized = shifted.normalize(with);
        Self {
            raw,
            shifted,
            normalized,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Deserialize, Serialize, PartialEq)]
pub struct Novelty(f64);

impl ScoreValue for Novelty {
    type Value = f64;

    fn value(&self) -> Self::Value {
        self.0
    }
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct NoveltyScore {
    pub raw: Raw<Novelty>,
    pub shifted: Shifted<Novelty>,
    pub normalized: Normalized<Novelty>,
}

impl NoveltyScore {
    pub fn new(raw: f64, baseline: f64, with: f64) -> Self {
        let raw = Raw::novelty(raw);
        let shifted = raw.shift(baseline);
        let normalized = shifted.normalize(with);
        Self {
            raw,
            shifted,
            normalized,
        }
    }
}

macro_rules! makeScoreType {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
            pub struct $name<T: ScoreValue>(T);

            impl<T: ScoreValue> ScoreType for $name<T> {}

            impl<T: ScoreValue> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: ScoreValue> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

makeScoreType!(Raw, Normalized, Shifted);

impl Raw<Fitness> {
    pub fn fitness(fitness: f64) -> Self {
        Self(Fitness(fitness))
    }
    pub fn shift(self, baseline: f64) -> Shifted<Fitness> {
        Shifted(Fitness(self.value() - baseline))
    }
}

impl Shifted<Fitness> {
    pub fn normalize(self, with: f64) -> Normalized<Fitness> {
        Normalized(Fitness(self.value() / with.max(1.0)))
    }
}

impl Raw<Novelty> {
    pub fn novelty(novelty: f64) -> Self {
        Self(Novelty(novelty))
    }
    pub fn shift(self, baseline: f64) -> Shifted<Novelty> {
        Shifted(Novelty(self.value() - baseline))
    }
}

impl Shifted<Novelty> {
    pub fn normalize(self, with: f64) -> Normalized<Novelty> {
        Normalized(Novelty(self.value() / with.max(1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::{Fitness, Normalized, Raw, Shifted};

    #[test]
    fn shift_raw() {
        let raw = Raw::fitness(1.0);

        let shifted = raw.shift(-2.0);

        assert_eq!(shifted, Shifted(Fitness(3.0)))
    }

    #[test]
    fn normalize_shifted() {
        let shifted = Shifted(Fitness(1.0));

        let normalized = shifted.normalize(2.0);

        assert_eq!(normalized, Normalized(Fitness(0.5)))
    }
}
