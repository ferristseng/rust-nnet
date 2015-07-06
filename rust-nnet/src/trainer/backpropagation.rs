use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

use num_cpus;
use threadpool::ScopedPool;
use trainer::util;
use trainer::util::TrainerState;
use prelude::{TrainerParameters, NeuralNet, TrainingSetMember, 
  NeuralNetTrainer, Layer, NeuralNetParameters};


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
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
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
/// calculated mean-squared-error, with an optional stopping condition 
/// based on the epoch. Weights are updated for each example in the training 
/// set.
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
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
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
  pub fn with_epoch_bound(
    nnet: &'a mut N, 
    tset: &'a [T], 
    mse: f64, 
    max: usize
  ) -> Self { 
    if mse <= 0f64 { panic!("target mse should be greater than 0") }

    SeqMSETrainer {
      nnet: nnet,
      tset: tset,
      epoch: 0,
      state: TrainerState::new::<_, N>(),
      mse_target: mse,
      max_epochs: max,
      tptype: PhantomData,
      nptype: PhantomData
    } 
  }
}

impl<'a, N, T, X, Y> NeuralNetTrainer for SeqMSETrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for SeqMSETrainer<'a, N, T, X, Y>
  where N : NeuralNet<Y>, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  type Item = (usize, f64);

  fn next(&mut self) -> Option<(usize, f64)> {
    if self.epoch == self.max_epochs {
      None
    } else {
      let mut sum = 0f64;

      for member in self.tset.iter() {
        util::update_state::<X, Y, _, _>(self.nnet, &mut self.state, member);
        util::update_weights(self.nnet, &self.state);
        
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


/// Back-propagation trainer where the stopping condition is based on a max 
/// number of epochs. Weights are updated at the end of each epoch.
///
pub struct BatchEpochTrainer<'a, N : 'a, T : 'a, X, Y> {
  tset: &'a [T],
  pool: ScopedPool<'a>,
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

impl<'a, N, T, X, Y> BatchEpochTrainer<'a, N, T, X, Y> 
  where N : NeuralNet<Y> + Clone, 
        T : TrainingSetMember, 
        X : TrainerParameters, 
        Y : NeuralNetParameters
{
  /// Creates a new trainer for a neural net, given a training set, where the 
  /// stopping condition is the number of epochs.
  ///
  #[inline(always)]
  pub fn new(nnet: &'a mut N, tset: &'a [T], epochs: usize) -> Self {
    let threads = num_cpus::get();

    BatchEpochTrainer {
      tset: tset,
      pool: ScopedPool::new(threads as u32),
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

impl<'a, N, T, X, Y> NeuralNetTrainer for BatchEpochTrainer<'a, N, T, X, Y> 
  where N : Send + NeuralNet<Y>, 
        T : Send + TrainingSetMember + Sync, 
        X : Send + TrainerParameters, 
        Y : Send + NeuralNetParameters
{ }

impl<'a, N, T, X, Y> Iterator for BatchEpochTrainer<'a, N, T, X, Y> 
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
      let (tx, rx) = mpsc::sync_channel(1); 

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

        self.pool.execute(move || {
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
        })
      } 

      // Average the accumulated states together into the current 
      // state.
      for state in rx.iter().take(self.threads) {
        self.state.combine(&state)
      }

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