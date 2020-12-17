use std::ops::{Deref, DerefMut};

use rand::prelude::SmallRng;
use serde::{Deserialize, Serialize};

use crate::{genes::IdGenerator, parameters::Parameters};

use self::scores::{FitnessScore, NoveltyScore, ScoreValue};
use self::{behavior::Behavior, genome::Genome};

pub mod behavior;
pub mod genome;
pub mod scores;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Individual {
    pub genome: Genome,
    pub age: usize,
    pub behavior: Option<Behavior>,
    pub fitness: Option<FitnessScore>,
    pub novelty: Option<NoveltyScore>,
}

impl Deref for Individual {
    type Target = Genome;

    fn deref(&self) -> &Self::Target {
        &self.genome
    }
}

impl DerefMut for Individual {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.genome
    }
}

impl Individual {
    pub fn initial(id_gen: &mut IdGenerator, parameters: &Parameters) -> Self {
        Self {
            genome: Genome::new(id_gen, parameters),
            age: 0,
            behavior: None,
            fitness: None,
            novelty: None,
        }
    }

    // score is combination of fitness & novelty
    pub fn score(&self) -> f64 {
        let novelty = self
            .novelty
            .as_ref()
            .map(|n| n.normalized.value())
            .unwrap_or(0.0);
        let fitness = self
            .fitness
            .as_ref()
            .map(|n| n.normalized.value())
            .unwrap_or(0.0);

        if novelty == 0.0 && fitness == 0.0 {
            return 0.0;
        }

        let (min, max) = if novelty < fitness {
            (novelty, fitness)
        } else {
            (fitness, novelty)
        };

        // ratio tells us what score is dominant in this genome
        let ratio = min / max / 2.0;

        // we weight the scores by their ratio, i.e. a genome that has a good fitness value is primarily weighted by that
        min * ratio + max * (1.0 - ratio)
    }

    // self is fitter if it has higher score or in case of equal score has fewer genes, i.e. less complexity
    pub fn is_fitter_than(&self, other: &Self) -> bool {
        let score_self = self.score();
        let score_other = other.score();

        score_self > score_other
            || ((score_self - score_other).abs() < f64::EPSILON
                && self.genome.len() < other.genome.len())
    }

    pub fn crossover(&self, other: &Self, rng: &mut SmallRng) -> Self {
        let (fitter, weaker) = if self.is_fitter_than(other) {
            (&self.genome, &other.genome)
        } else {
            (&other.genome, &self.genome)
        };

        Individual {
            genome: fitter.cross_in(weaker, rng),
            age: 0,
            behavior: None,
            fitness: None,
            novelty: None,
        }
    }
}
