mod backpropagation_;
mod util;

/// Implementation of backpropagation trainers.
///
pub mod backpropagation {
  pub use trainer::backpropagation_::{
    SeqEpochTrainer, 
    SeqErrorAverageTrainer,
    BatchEpochTrainer
  };

  /// Multithreaded implementations of backpropagation trainers.
  ///
  pub mod parallel {
    pub use trainer::backpropagation_::BatchEpochTrainerParallel 
         as BatchEpochTrainer;
  }
}