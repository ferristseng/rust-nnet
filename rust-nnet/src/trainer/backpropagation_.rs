use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

use num_cpus;
use scoped_threadpool::Pool;
use prelude::*;
use trainer::util;
use trainer::util::TrainerState;


/// Back-propagation trainer where the stopping criteria is bounded by the 
/// epoch. Weights are updated for each example in the training set.
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
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// max number of epochs is set to `::std::usize::MAX`.
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T]) -> Self {
    Self::with_epochs(nnet, tset, ::std::usize::MAX)
  }

  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn with_epochs(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    SeqEpochTrainer {
      nnet: nnet,
      tset: tset,
      state: TrainerState::new::<_, N>(),
      epoch: 0,
      max_epochs: epochs,
      tptype: PhantomData,
      nptype: PhantomData
    }
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for SeqEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y>  Iterator for SeqEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let epoch = self.epoch;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &self.state);
      }

      self.epoch += 1;

      Some(epoch)
    }
  }
}


/// Back-propagation trainer where the stopping condition is primarily the 
/// calculated average error, with an optional stopping condition based on the 
/// epoch. Weights are updated for each example in the training set.
///
pub struct SeqErrorAverageTrainer<'a, N : 'a, T : 'a, X, Y> {
  nnet: &'a mut N,
  tset: &'a [T],
  epoch: usize,
  state: TrainerState,
  err_target: f64,
  max_epochs: usize,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> SeqErrorAverageTrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParametersWithErrorFunction, 
        Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set and target 
  /// `err`. By default, the max number of epochs the trainer can run is 
  /// the max value for `usize`.
  ///
  /// # Panics
  ///
  /// When `err` is less than or equal to 0.
  ///
  #[inline(always)] 
  pub fn new(nnet: &'a mut N, tset: &'a [T], err: f64) -> Self {
    Self::with_epoch_bound(nnet, tset, err, ::std::usize::MAX)
  }

  /// Creates a new trainer for a neural net, given a training set and target 
  /// `err` and target max epoch as an alternate stopping condition.
  ///
  /// # Panics
  /// 
  /// When `err` is less than or equal to 0.
  ///
  #[inline(always)] 
  pub fn with_epoch_bound(
    nnet: &'a mut N, 
    tset: &'a [T], 
    err: f64, 
    max: usize
  ) -> Self { 
    if err <= 0f64 { panic!("target err should be greater than 0") }

    SeqErrorAverageTrainer {
      nnet: nnet,
      tset: tset,
      epoch: 0,
      state: TrainerState::new::<_, N>(),
      err_target: err,
      max_epochs: max,
      tptype: PhantomData,
      nptype: PhantomData
    } 
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for SeqErrorAverageTrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParametersWithErrorFunction, 
        Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for SeqErrorAverageTrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParametersWithErrorFunction, 
        Y : NeuralNetParameters
{
  type Item = (usize, f64);

  fn next(&mut self) -> Option<(usize, f64)> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let mut err = 0f64;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &self.state);
        
        let exp = member.expected();
        let act = self.nnet.layer(Layer::Output);

        err += X::ErrorFunction::error(act.iter(), exp.iter());
      }

      let avg = err / self.tset.len() as f64;
      let ret = Some((self.epoch, avg));

      if avg <= self.err_target { 
        self.max_epochs = self.epoch;
      } else {
        self.epoch += 1;
      }

      ret
    }
  }
}


/// Back-propagation trainer where the stopping condition is based on a max 
/// number of epochs. Weights are updated at the end of each epoch.
///
pub struct BatchEpochTrainer<'a, N : 'a, T : 'a, X, Y> {
  nnet: &'a mut N,
  tset: &'a [T],
  state: TrainerState,
  epoch: usize,
  max_epochs: usize,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    BatchEpochTrainer {
      nnet: nnet,
      tset: tset,
      state: TrainerState::new::<_, N>(),
      epoch: 0,
      max_epochs: epochs,
      tptype: PhantomData,
      nptype: PhantomData
    }
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let epoch = self.epoch;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
      }

      util::update_weights(self.nnet, &self.state);

      self.epoch += 1;

      Some(epoch)
    }
  }
}


/// (Parallelized) Back-propagation trainer where the stopping condition 
/// is based on a max number of epochs. Weights are updated at the end 
/// of each epoch.
///
pub struct BatchEpochTrainerParallel<'a, N : 'a, T : 'a, X, Y> {
  tset: &'a [T],
  pool: Pool,
  size: usize,
  state: TrainerState,
  epoch: usize,
  threads: usize,
  max_epochs: usize,
  owned_nnet: Arc<Mutex<N>>,
  borrowed_nnet: &'a mut N,
  tptype: PhantomData<X>,
  nptype: PhantomData<Y>
}

impl<'a, N, T, X, Y> BatchEpochTrainerParallel<'a, N, T, X, Y> 
  where N : NeuralNet<Y> + Clone, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// max number of epochs is set to `::std::usize::MAX`.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T]) -> Self {
    Self::with_epochs(nnet, tset, ::std::usize::MAX)
  }

  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn with_epochs(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    let threads = num_cpus::get();

    BatchEpochTrainerParallel {
      tset: tset,
      pool: Pool::new(threads as u32),
      size: tset.len() / threads,
      state: TrainerState::new::<_, N>(),
      epoch: 0,
      threads: threads, 
      max_epochs: epochs,
      owned_nnet: Arc::new(Mutex::new(nnet.clone())),
      borrowed_nnet: nnet,
      tptype: PhantomData,
      nptype: PhantomData
    }
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for BatchEpochTrainerParallel<'a, N, T, X, Y> 
  where N : Send + NeuralNet<Y>, 
        T : Send + TrainingSetMember + Sync, 
        X : Send + TrainerParameters, 
        Y : Send + NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for BatchEpochTrainerParallel<'a, N, T, X, Y> 
  where N : Send + NeuralNet<Y>, 
        T : Send + TrainingSetMember + Sync, 
        X : Send + TrainerParameters, 
        Y : Send + NeuralNetParameters
{
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let epoch = self.epoch;
      let (tx, rx) = mpsc::channel(); 

      // Threaded implementation. Launch a thread to work on 
      // a specific section of the training set for the current epoch.
      // Each thread has its own copy of the training state, and modifies 
      // that local copy. 
      for i in 0..self.threads {
        let send = tx.clone();
        let start = i * self.size;
        let state = self.state.clone();
        let tslice = &self.tset[start..start + self.size];
        let lock = self.owned_nnet.clone();

        self.pool.scoped(|scope| scope.execute(move || {
          let mut state = state;

          for member in tslice.iter() {
            match lock.lock() {
              Ok(mut nnet) => {
                let nnet: &mut N = &mut nnet;
                util::update_state::<X, Y, _, _>(nnet, &mut state, member);
              }
              Err(e) => warn!("error locking nnet: {:?}", e)
            };
          }
          
          match send.send(state) {
            Ok(_) => (),
            Err(e) => warn!("error sending: {:?}", e) 
          }
        }))
      } 

      // Average the accumulated states together into the current 
      // state.
      self.state.combine(rx.iter().take(self.threads));

      // Update the weights of the neural nets. The owned copy 
      // of the neural net needs to be updated in sync.
      match self.owned_nnet.lock() {
        Ok(mut nnet) => {
          let nnet: &mut N = &mut nnet;
          util::update_weights(nnet, &self.state);
          util::update_weights(self.borrowed_nnet, &self.state);
        }
        Err(e) => warn!("error locking nnet: {:?}", e)
      }

      self.epoch += 1;

      Some(epoch)
    }
  }
}