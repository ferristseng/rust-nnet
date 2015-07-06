mod backpropagation_;
mod util;

pub mod backpropagation {
  pub use trainer::backpropagation_::{
    SeqEpochTrainer, 
    SeqMSETrainer,
    BatchEpochTrainer
  };

  pub mod parallel {
    pub use trainer::backpropagation_::BatchEpochTrainerParallel 
         as BatchEpochTrainer;
  }
}