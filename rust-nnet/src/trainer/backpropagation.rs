use std::marker::PhantomData;

use trainer::util;
use trainer::util::TrainerState;
use prelude::{TrainerParameters, NeuralNet, TrainingSetMember, 
  NeuralNetTrainer, Layer};


/// Back-propagation trainer where the stopping criteria is bounded by the epoch.
pub struct IncrementalEpochTrainer<'a, N : 'a, T : 'a, P> {
  nnet: &'a mut N,
  tset: &'a [T],
  state: TrainerState,
  epochs: usize,
  max_epochs: usize,
  ptype: PhantomData<P>
}

impl<'a, N, T, P> IncrementalEpochTrainer<'a, N, T, P> 
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    let state = TrainerState::new(nnet);

    IncrementalEpochTrainer {
      nnet: nnet,
      tset: tset,
      state: state,
      epochs: 0,
      max_epochs: epochs,
      ptype: PhantomData
    }
  }
}

impl<'a, N, T, P> NeuralNetTrainer for IncrementalEpochTrainer<'a, N, T, P>
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{ }

impl<'a, N, T, P> Iterator for IncrementalEpochTrainer<'a, N, T, P>
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epochs == self.max_epochs {
      None
    } else {
      for member in self.tset.iter() {
        util::update_state::<P, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &mut self.state);
      }

      self.epochs += 1;

      Some(self.epochs)
    }
  }
}


/// Back-propagation trainer where the stopping condition is primarily the 
/// calculated mean-squared-error, with an optional stopping condition 
/// based on the epoch.
pub struct IncrementalMSETrainer<'a, N : 'a, T : 'a, P> {
  nnet: &'a mut N,
  tset: &'a [T],
  epoch: usize,
  state: TrainerState,
  mse_target: f64,
  max_epochs: usize,
  ptype: PhantomData<P>
}

impl<'a, N, T, P> IncrementalMSETrainer<'a, N, T, P>
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{
  /// Creates a new trainer for a neural net, given a training set and target 
  /// `mse`. By default, the max number of epochs the trainer can run is 
  /// the max value for `usize`.
  ///
  /// # Panics
  ///
  /// When `mse` is less than or equal to 0.
  ///
  #[inline(always)] 
  pub fn new(nnet: &'a mut N, tset: &'a [T], mse: f64) -> Self {
    IncrementalMSETrainer::with_epoch_bound(nnet, tset, mse, ::std::usize::MAX)
  }

  /// Creates a new trainer for a neural net, given a training set and target 
  /// `mse` and target max epoch as an alternate stopping condition.
  ///
  /// # Panics
  /// 
  /// When `mse` is less than or equal to 0.
  ///
  #[inline(always)] 
  pub fn with_epoch_bound(nnet: &'a mut N, tset: &'a [T], mse: f64, max: usize) -> Self { 
    if mse <= 0f64 { panic!("target mse should be greater than 0") }

    let state = TrainerState::new(nnet);

    IncrementalMSETrainer {
      nnet: nnet,
      tset: tset,
      epoch: 0,
      state: state,
      mse_target: mse,
      max_epochs: max,
      ptype: PhantomData
    } 
  }
}

impl<'a, N, T, P> NeuralNetTrainer for IncrementalMSETrainer<'a, N, T, P>
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{ }

impl<'a, N, T, P> Iterator for IncrementalMSETrainer<'a, N, T, P>
  where N : NeuralNet, T : TrainingSetMember, P : TrainerParameters
{
  type Item = (usize, f64);

  fn next(&mut self) -> Option<(usize, f64)> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let mut sum = 0f64;

      for member in self.tset.iter() {
        util::update_state::<P, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &mut self.state);
        
        let exp = member.expected();
        let act = self.nnet.layer(Layer::Output);

        sum += util::mse(act.iter(), exp.iter());
      }

      let mse = sum / self.tset.len() as f64;
      let ret = Some((self.epoch, mse));

      if mse <= self.mse_target { 
        self.epoch = self.max_epochs;
      } else {
        self.epoch += 1;
      }

      ret
    }
  }
}