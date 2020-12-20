use std::time::{Instant, SystemTime};

use crate::{
    individual::Individual, population::Population, utility::statistics::Statistics, Neat,
};

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use self::{evaluation::Evaluation, progress::Progress};

pub mod evaluation;
pub mod progress;

pub struct Runtime<'a> {
    neat: &'a Neat,
    population: Population,
    statistics: Statistics,
}

impl<'a> Runtime<'a> {
    pub fn new(neat: &'a Neat) -> Self {
        Self {
            neat,
            population: Population::new(&neat.parameters),
            statistics: Statistics::default(),
        }
    }

    fn generate_progress(&self) -> Vec<Progress> {
        let progress_fn = &self.neat.progress_function;

        // apply progress function to every individual
        self.population
            .individuals()
            .par_iter()
            .map(progress_fn)
            .collect::<Vec<Progress>>()
    }

    fn check_for_solution(&self, progress: &[Progress]) -> Option<Individual> {
        progress
            .iter()
            .filter_map(|p| p.is_solution())
            .cloned()
            .next()
    }
}

impl<'a> Iterator for Runtime<'a> {
    type Item = Evaluation;

    fn next(&mut self) -> Option<Self::Item> {
        self.statistics.time_stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let now = Instant::now();

        // generate progress by running progress function for every individual
        let progress = self.generate_progress();

        self.statistics.num_generation += 1;
        self.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        if let Some(winner) = self.check_for_solution(&progress) {
            Some(Evaluation::Solution(winner))
        } else {
            self.statistics.population = self
                .population
                .next_generation(&self.neat.parameters, &progress);

            Some(Evaluation::Progress(self.statistics.clone()))
        }
    }
}
