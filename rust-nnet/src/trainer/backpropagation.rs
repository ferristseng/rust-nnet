  use std::marker::PhantomData;


  use trainer::util;
  use trainer::util::TrainerState;
  use prelude::{TrainerParameters, NeuralNet, TrainingSetMember, NeuralNetTrainer};


  /// Back-propagation trainer where the stopping criteria is based on Epoch.
  pub struct SequentialEpochTrainer<P> { 
    epochs: usize,
    ptype : PhantomData<P>
  }

  impl<P> NeuralNetTrainer for SequentialEpochTrainer<P>
    where P : TrainerParameters<SequentialEpochTrainer<P>>
  {
    fn train<N, T>(&self, nn: &mut N, ex: &[T]) 
      where N : NeuralNet, T : TrainingSetMember 
    {
      let mut state = TrainerState::new(nn);

      for _ in (0..self.epochs) {
        for member in ex.iter() {
          util::update_state::<P, Self, N, T>(self, nn, &mut state, member);
          util::update_weights(nn, &mut state);
        }
      }
    }
  }

  impl<P> SequentialEpochTrainer<P> 
    where P : TrainerParameters<SequentialEpochTrainer<P>>
  {
    pub fn new(epochs: usize) -> SequentialEpochTrainer<P> {
      SequentialEpochTrainer {
        epochs: epochs,
        ptype: PhantomData
      }
    }
  }