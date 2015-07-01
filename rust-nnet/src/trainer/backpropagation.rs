use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

use num_cpus;
use threadpool::ScopedPool;
use trainer::util;
use trainer::util::TrainerState;
use prelude::{TrainerParameters, NeuralNet, TrainingSetMember, 
  NeuralNetTrainer, Layer, NeuralNetParameters};


/// Back-propagation trainer where the stopping criteria is bounded by the epoch.
/// Weights are updated for each example in the training set.
///
pub struct SeqEpochTrainer<'a, N : 'a, T : 'a, X, Y> {
  nnet: &'a mut N,
  tset: &'a [T],
  state: TrainerState,
  epoch: usize,
  max_epochs: usize,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> SeqEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    let state = TrainerState::new(nnet);

    SeqEpochTrainer {
      nnet: nnet,
      tset: tset,
      state: state,
      epoch: 0,
      max_epochs: epochs,
      tptype: PhantomData,
      nptype: PhantomData
    }
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for SeqEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y>  Iterator for SeqEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let epoch = self.epoch;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &mut self.state);
      }

      self.epoch += 1;

      Some(epoch)
    }
  }
}


/// Back-propagation trainer where the stopping condition is primarily the 
/// calculated mean-squared-error, with an optional stopping condition 
/// based on the epoch. Weights are updated for each example in the training set.
///
pub struct SeqMSETrainer<'a, N : 'a, T : 'a, X, Y> {
  nnet: &'a mut N,
  tset: &'a [T],
  epoch: usize,
  state: TrainerState,
  mse_target: f64,
  max_epochs: usize,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> SeqMSETrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
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
    SeqMSETrainer::with_epoch_bound(nnet, tset, mse, ::std::usize::MAX)
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

    SeqMSETrainer {
      nnet: nnet,
      tset: tset,
      epoch: 0,
      state: state,
      mse_target: mse,
      max_epochs: max,
      tptype: PhantomData,
      nptype: PhantomData
    } 
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for SeqMSETrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for SeqMSETrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{
  type Item = (usize, f64);

  fn next(&mut self) -> Option<(usize, f64)> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let mut sum = 0f64;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &mut self.state);
        
        let exp = member.expected();
        let act = self.nnet.layer(Layer::Output);

        sum += util::mse(act.iter(), exp.iter());
      }

      let mse = sum / self.tset.len() as f64;
      let ret = Some((self.epoch, mse));

      if mse <= self.mse_target { 
        self.max_epochs = self.epoch;
      } else {
        self.epoch += 1;
      }

      ret
    }
  }
}


/// Back-propagation trainer where the stopping condition is based on a max number 
/// of epochs. Weights are updated at the end of each epoch.
///
pub struct BatchEpochTrainer<'a, N : 'a, T : 'a, X, Y> {
  nnet: &'a mut N,
  tset: &'a [T],
  pool: ScopedPool<'a>,
  state: Mutex<TrainerState>,
  epoch: usize,
  max_epochs: usize,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, T : TrainingSetMember, X : TrainerParameters, Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    let state = TrainerState::new(nnet);

    BatchEpochTrainer {
      nnet: nnet,
      tset: tset,
      pool: ScopedPool::new(num_cpus::get() as u32),
      state: Mutex::new(state),
      epoch: 0,
      max_epochs: epochs,
      tptype: PhantomData,
      nptype: PhantomData
    }
  }
}

impl<'a, N, T, X, Y>  NeuralNetTrainer for BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y> + Send + ::std::fmt::Debug, 
        T : TrainingSetMember + Sync + ::std::fmt::Debug + Send, 
        X : TrainerParameters + Send, 
        Y : NeuralNetParameters + Send + ::std::fmt::Debug
{ }

impl<'a, N, T, X, Y>  Iterator for BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y> + Send + ::std::fmt::Debug, 
        T : TrainingSetMember + Sync + ::std::fmt::Debug + Send, 
        X : TrainerParameters + Send, 
        Y : NeuralNetParameters + Send + ::std::fmt::Debug
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let threads = num_cpus::get();
      let size = self.tset.len() / threads;
      let epoch = self.epoch;

      for i in 0..threads {
        let start = i * size;
        let tslice = &self.tset[start..start + size];
        self.pool.execute(move || {
          for member in tslice.iter() {
            //let state = state.lock().unwrap();
            //util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
            //println!("{:?}", self.nnet);
          } 
        });
      } 
      
      //util::update_weights(self.nnet, &mut self.state);

      self.epoch += 1;

      Some(epoch)
    }
  }
}