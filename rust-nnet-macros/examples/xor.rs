extern crate nnet;
#[macro_use(ffnn)] extern crate nnet_macros;

use nnet::trainer::backpropagation::IncrementalMSETrainer;
use nnet::params::{TanhNeuralNet, Default};
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, 
  LearningRate};


ffnn!(XORNeuralNet, 2, 3, 1);


struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  fn momentum() -> f64 { 0.8f64 }
}

impl<T> LearningRate<T> for MyTrainerParams where T : NeuralNetTrainer {
  fn lrate(_: &T) -> f64 { 0.3f64 }
}


fn main() {
  let xor: [(&[f64], &[f64]); 4] = [
    (&[0f64, 0f64], &[0f64]),
    (&[0f64, 1f64], &[1f64]),
    (&[1f64, 0f64], &[1f64]),
    (&[1f64, 1f64], &[0f64])
  ];

  let mut nn: XORNeuralNet<TanhNeuralNet> = XORNeuralNet::new();
  let trainer: IncrementalMSETrainer<MyTrainerParams> = 
    IncrementalMSETrainer::with_epoch_bound(0.01, 5000);

  trainer.train(&mut nn, &xor);

  for ex in xor.iter() {
    nn.predict(ex.0);
    println!("{:?} - prediction = {:?}", ex.0, nn.output);
  }
}