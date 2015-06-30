use std::marker::PhantomData;

use trainer::util;
use trainer::util::TrainerState;
use prelude::{TrainerParameters, NeuralNet, TrainingSetMember, 
  NeuralNetTrainer, Layer};


/// Back-propagation trainer where the stopping criteria is based on Epoch.
pub struct IncrementalEpochTrainer<P> { 
  epochs: usize,
  ptype : PhantomData<P>
}

impl<P> NeuralNetTrainer for IncrementalEpochTrainer<P>
  where P : TrainerParameters<IncrementalEpochTrainer<P>>
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

impl<P> IncrementalEpochTrainer<P> 
  where P : TrainerParameters<IncrementalEpochTrainer<P>>
{
  pub fn new(epochs: usize) -> IncrementalEpochTrainer<P> {
    IncrementalEpochTrainer {
      epochs: epochs,
      ptype: PhantomData
    }
  }
}


pub struct IncrementalMSETrainer<P> {
  mse_target: f64,
  max_epochs: Option<usize>,
  ptype: PhantomData<P>,
}

impl<P> NeuralNetTrainer for IncrementalMSETrainer<P>
  where P : TrainerParameters<IncrementalMSETrainer<P>>
{
  fn train<N, T>(&self, nn: &mut N, ex: &[T])
    where N : NeuralNet, T : TrainingSetMember
  {
    let mut state = TrainerState::new(nn);
    let mut epoch = 0;

    loop {
      let mut sum = 0f64;

      for member in ex.iter() {
        util::update_state::<P, Self, N, T>(self, nn, &mut state, member);
        util::update_weights(nn, &mut state);
        
        let exp = member.expected();
        let act = nn.layer(Layer::Output);

        sum += util::mse(act.iter(), exp.iter());
      }

      if (sum / ex.len() as f64) <= self.mse_target ||  
        self.max_epochs.map(|e| e == epoch).unwrap_or(false)
      { 
        break 
      }

      epoch += 1;
    }
  }
}


impl<P> IncrementalMSETrainer<P>
  where P : TrainerParameters<IncrementalMSETrainer<P>>
{
  pub fn new(mse: f64) -> IncrementalMSETrainer<P> {
    if mse <= 0f64 { panic!("target mse should be greater than 0") }

    IncrementalMSETrainer {
      mse_target: mse,
      max_epochs: None,
      ptype: PhantomData
    } 
  }

  pub fn with_epoch_bound(mse: f64, max_epochs: usize) -> IncrementalMSETrainer<P> {
    if mse <= 0f64 { panic!("target mse should be greater than 0") }

    IncrementalMSETrainer {
      mse_target: mse,
      max_epochs: Some(max_epochs),
      ptype: PhantomData
    } 
  }
}