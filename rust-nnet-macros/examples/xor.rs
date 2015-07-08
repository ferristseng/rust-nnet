extern crate time;
extern crate nnet;
#[macro_use(ffnn)] extern crate nnet_macros;

use time::PreciseTime;
use nnet::trainer::backpropagation::*;
use nnet::params::{TanhNeuralNet, LogisticNeuralNet};
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, 
  Layer, LearningRate};


ffnn!(XORNeuralNet, 2, 3, 1);


struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  #[inline(always)] fn momentum() -> f64 { 0.4f64 }
}

impl LearningRate for MyTrainerParams {
  #[inline(always)] fn lrate() -> f64 { 0.1f64 }
}


fn main() {
  let xor: [(&[f64], &[f64]); 4] = [
    (&[0f64, 0f64], &[0f64]),
    (&[0f64, 1f64], &[1f64]),
    (&[1f64, 0f64], &[1f64]),
    (&[1f64, 1f64], &[0f64])
  ];

  let mut nn: XORNeuralNet<TanhNeuralNet> = XORNeuralNet::new();
  let start = PreciseTime::now();
  
  // Train sequentially, using a set number of epochs.
  SeqEpochTrainer::<_, _, MyTrainerParams, _>::new(&mut nn, &xor, 100000).finish();

  println!("took = {:?} ms", start.to(PreciseTime::now()).num_milliseconds());

  // Check to see if we learned anything!
  for ex in xor.iter() {
    nn.predict(ex.0);
    println!("{:?} - prediction = {:?}", ex.0, nn.layer(Layer::Output));
  }
}
