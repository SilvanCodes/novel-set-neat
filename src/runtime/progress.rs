use crate::individual::{
    behavior::Behavior,
    scores::{Fitness, Raw},
    Individual,
};

#[derive(Debug)]
pub enum Progress {
    Empty,
    Novelty(Behavior),
    Status(Raw<Fitness>, Behavior),
    Solution(Option<Raw<Fitness>>, Option<Behavior>, Box<Individual>),
}

impl Progress {
    pub fn new(fitness: f64, behavior: Vec<f64>) -> Self {
        Progress::Status(Raw::fitness(fitness), Behavior(behavior))
    }

    pub fn empty() -> Self {
        Progress::Empty
    }

    pub fn solved(self, solution: Individual) -> Self {
        match self {
            Progress::Novelty(behavior) => {
                Progress::Solution(None, Some(behavior), Box::new(solution))
            }
            Progress::Status(fitness, behavior) => {
                Progress::Solution(Some(fitness), Some(behavior), Box::new(solution))
            }
            Progress::Solution(fitness, behavior, _) => {
                Progress::Solution(fitness, behavior, Box::new(solution))
            }
            Progress::Empty => Progress::Solution(None, None, Box::new(solution)),
        }
    }

    pub fn novelty(behavior: Vec<f64>) -> Self {
        Self::Novelty(Behavior(behavior))
    }

    pub fn behavior(&self) -> Option<&Behavior> {
        match self {
            Progress::Status(_, behavior) => Some(behavior),
            Progress::Solution(_, behavior, _) => behavior.as_ref(),
            Progress::Novelty(behavior) => Some(behavior),
            Progress::Empty => None,
        }
    }

    pub fn raw_fitness(&self) -> Option<Raw<Fitness>> {
        match *self {
            Progress::Status(fitness, _) => Some(fitness),
            Progress::Solution(fitness, _, _) => fitness,
            Progress::Novelty(_) => None,
            Progress::Empty => None,
        }
    }

    pub fn is_solution(&self) -> Option<&Individual> {
        match self {
            Progress::Solution(_, _, individual) => Some(individual),
            _ => None,
        }
    }
}
