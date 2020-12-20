pub use individual::Individual;
use parameters::Parameters;
pub use runtime::{evaluation::Evaluation, progress::Progress, Runtime};

mod genes;
mod individual;
mod parameters;
mod population;
mod runtime;
pub mod utility;

pub struct Neat {
    pub parameters: Parameters,
    progress_function: Box<dyn Fn(&Individual) -> Progress + Send + Sync>,
}

// public API
impl Neat {
    pub fn new(
        path: &str,
        progress_function: Box<dyn Fn(&Individual) -> Progress + Send + Sync>,
    ) -> Self {
        Neat {
            parameters: Parameters::new(path).unwrap(),
            progress_function,
        }
    }

    pub fn run(&self) -> Runtime {
        Runtime::new(&self)
    }
}
