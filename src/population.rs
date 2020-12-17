use std::time::Instant;

use rand::prelude::SliceRandom;

use crate::{
    genes::IdGenerator,
    individual::{
        behavior::{Behavior, Behaviors},
        scores::{Fitness, FitnessScore, NoveltyScore, Raw, ScoreValue},
        Individual,
    },
    parameters::Parameters,
    rng::NeatRng,
    runtime::progress::Progress,
    statistics::PopulationStatistics,
};

pub struct Population {
    individuals: Vec<Individual>,
    archive: Vec<Individual>,
    population_statistics: PopulationStatistics,
    rng: NeatRng,
    id_gen: IdGenerator,
}

impl Population {
    pub fn new(parameters: &Parameters) -> Self {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        // generate genome with initial ids for structure
        let initial_individual = Individual::initial(&mut id_gen, parameters);

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.seed,
            parameters.mutation.weights.perturbation_std_dev,
        );

        let mut individuals = Vec::new();

        // generate initial, mutated individuals
        for _ in 0..parameters.setup.population_size {
            let mut other_genome = initial_individual.clone();
            other_genome.init(&mut rng, parameters);
            other_genome.mutate(&mut rng, &mut id_gen, parameters);
            individuals.push(other_genome);
        }

        Population {
            individuals,
            archive: Vec::new(),
            rng,
            id_gen,
            population_statistics: PopulationStatistics::default(),
        }
    }

    pub fn individuals(&self) -> &Vec<Individual> {
        &self.individuals
    }

    fn generate_offspring(&mut self, parameters: &Parameters) {
        let now = Instant::now();

        let partners = self.individuals.as_slice();

        let mut offsprings = Vec::new();

        // generate as many offspring as population size allows
        for parent in self
            .individuals
            .iter()
            .cycle()
            .take(parameters.setup.population_size - self.individuals.len())
        {
            let mut offspring = parent.crossover(
                partners
                    .choose(&mut self.rng.small)
                    .expect("could not select random partner"),
                &mut self.rng.small,
            );
            offspring.mutate(&mut self.rng, &mut self.id_gen, parameters);
            offsprings.push(offspring);
        }

        self.individuals.extend(offsprings.into_iter());

        // mutate entire population here ?

        self.population_statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }

    fn calculate_novelty(&mut self, parameters: &Parameters) {
        let behaviors: Behaviors = self
            .individuals
            .iter()
            .flat_map(|individual| individual.behavior.as_ref())
            .chain(
                self.archive
                    .iter()
                    .flat_map(|archived_individual| archived_individual.behavior.as_ref()),
            )
            .collect::<Vec<&Behavior>>()
            .into();

        let behavior_count = behaviors.len() as f64;

        let raw_novelties = behaviors.compute_novelty(parameters.novelty.nearest_neighbors);

        let most_novel = raw_novelties
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("could not compare floats"))
            .map(|(index, _)| index)
            .expect("failed finding most novel");

        // add most novel individual to archive if it resides in population
        if most_novel < self.individuals.len() - 1 {
            self.archive.push(self.individuals[most_novel].clone());
        }

        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        // analyse raw novelty values
        for &novelty in &raw_novelties {
            if novelty > raw_maximum {
                raw_maximum = novelty;
            }
            if novelty < raw_minimum {
                raw_minimum = novelty;
            }
            raw_sum += novelty;
        }

        let raw_minimum = Raw::novelty(raw_minimum);
        let raw_average = Raw::novelty(raw_sum / behavior_count);
        let raw_maximum = Raw::novelty(raw_maximum);

        let baseline = raw_minimum.value();

        let shifted_minimum = raw_minimum.shift(baseline);
        let shifted_average = raw_average.shift(baseline);
        let shifted_maximum = raw_maximum.shift(baseline);

        let with = shifted_maximum.value();

        let normalized_minimum = shifted_minimum.normalize(with);
        let normalized_average = shifted_average.normalize(with);
        let normalized_maximum = shifted_maximum.normalize(with);

        for (index, individual) in self.individuals.iter_mut().enumerate() {
            individual.novelty = Some(NoveltyScore::new(raw_novelties[index], baseline, with));
        }

        self.population_statistics.novelty.raw_maximum = raw_maximum.value();
        self.population_statistics.novelty.raw_minimum = raw_minimum.value();
        self.population_statistics.novelty.raw_average = raw_average.value();

        self.population_statistics.novelty.shifted_maximum = shifted_maximum.value();
        self.population_statistics.novelty.shifted_minimum = shifted_minimum.value();
        self.population_statistics.novelty.shifted_average = shifted_average.value();

        self.population_statistics.novelty.normalized_maximum = normalized_maximum.value();
        self.population_statistics.novelty.normalized_minimum = normalized_minimum.value();
        self.population_statistics.novelty.normalized_average = normalized_average.value();
    }

    fn assign_behavior(&mut self, progress: &[Progress]) {
        let behaviors: Vec<(usize, &Behavior)> = progress
            .iter()
            .enumerate()
            .flat_map(|(index, progress)| progress.behavior().map(|raw| (index, raw)))
            .collect();

        if behaviors.is_empty() {
            return;
        }

        for (index, behavior) in behaviors {
            self.individuals[index].behavior = Some(behavior.clone());
        }
    }

    fn assign_fitness(&mut self, progress: &[Progress]) {
        let fitnesses: Vec<(usize, Raw<Fitness>)> = progress
            .iter()
            .enumerate()
            .flat_map(|(index, progress)| progress.raw_fitness().map(|raw| (index, raw)))
            .collect();

        if fitnesses.is_empty() {
            return;
        }

        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        // analyse raw fitness values
        for (_, raw_fitness) in &fitnesses {
            if raw_fitness.value() > raw_maximum {
                raw_maximum = raw_fitness.value();
            }
            if raw_fitness.value() < raw_minimum {
                raw_minimum = raw_fitness.value();
            }
            raw_sum += raw_fitness.value();
        }

        let raw_minimum = Raw::fitness(raw_minimum);
        let raw_average = Raw::fitness(raw_sum / fitnesses.len() as f64);
        let raw_maximum = Raw::fitness(raw_maximum);

        let baseline = raw_minimum.value();

        let shifted_minimum = raw_minimum.shift(baseline);
        let shifted_average = raw_average.shift(baseline);
        let shifted_maximum = raw_maximum.shift(baseline);

        let with = shifted_maximum.value();

        let normalized_minimum = shifted_minimum.normalize(with);
        let normalized_average = shifted_average.normalize(with);
        let normalized_maximum = shifted_maximum.normalize(with);

        // shift and normalize fitness
        for (index, raw_fitness) in fitnesses {
            self.individuals[index].fitness =
                Some(FitnessScore::new(raw_fitness.value(), baseline, with));
        }

        self.population_statistics.fitness.raw_maximum = raw_maximum.value();
        self.population_statistics.fitness.raw_minimum = raw_minimum.value();
        self.population_statistics.fitness.raw_average = raw_average.value();

        self.population_statistics.fitness.shifted_maximum = shifted_maximum.value();
        self.population_statistics.fitness.shifted_minimum = shifted_minimum.value();
        self.population_statistics.fitness.shifted_average = shifted_average.value();

        self.population_statistics.fitness.normalized_maximum = normalized_maximum.value();
        self.population_statistics.fitness.normalized_minimum = normalized_minimum.value();
        self.population_statistics.fitness.normalized_average = normalized_average.value();
    }

    fn top_fitness_performer(&mut self) -> Individual {
        self.individuals.sort_by(|individual_0, individual_1| {
            individual_1
                .fitness
                .as_ref()
                .map(|f| f.normalized.value())
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(
                    &individual_0
                        .fitness
                        .as_ref()
                        .map(|f| f.normalized.value())
                        .unwrap_or(f64::NEG_INFINITY),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "failed to compare score {} and score {}",
                        individual_0.score(),
                        &individual_1.score()
                    )
                })
        });

        self.individuals
            .first()
            .expect("individuals are empty!")
            .clone()
    }

    pub fn next_generation(
        &mut self,
        parameters: &Parameters,
        progress: &[Progress],
    ) -> PopulationStatistics {
        self.assign_fitness(progress);
        self.assign_behavior(progress);
        self.calculate_novelty(parameters);

        // sort individuals by their score (descending, i.e. highest score first)
        self.individuals.sort_by(|individual_0, individual_1| {
            individual_1
                .score()
                .partial_cmp(&individual_0.score())
                .unwrap_or_else(|| {
                    panic!(
                        "failed to compare score {} and score {}",
                        individual_0.score(),
                        individual_1.score()
                    )
                })
        });

        // add best scoring individual to archive ?? (should be most novel ?)
        // self.archive.push(self.individuals[0].clone());

        // remove any individual that does not survive
        self.individuals.truncate(
            (parameters.setup.population_size as f64 * parameters.reproduction.surviving).ceil()
                as usize,
        );

        // increment age of surviving individuals
        for individual in &mut self.individuals {
            individual.age += 1;
        }

        // reproduce from surviving individuals
        self.generate_offspring(parameters);

        // return some statistics
        self.gather_statistics()
    }

    fn gather_statistics(&mut self) -> PopulationStatistics {
        self.population_statistics.top_performer = self.top_fitness_performer();

        // determine maximum age
        self.population_statistics.age_maximum = self
            .individuals
            .iter()
            .map(|individual| individual.age)
            .max()
            .expect("cant find max age");

        // determine average age
        self.population_statistics.age_average = self
            .individuals
            .iter()
            .map(|individual| individual.age as f64)
            .sum::<f64>()
            / self.individuals.len() as f64;

        self.population_statistics.clone()
    }
}
