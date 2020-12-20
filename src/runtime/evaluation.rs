use crate::{individual::Individual, utility::statistics::Statistics};

pub enum Evaluation {
    Progress(Statistics),
    Solution(Individual),
}
