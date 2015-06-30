extern crate nnet;
#[macro_use(ffnn)] extern crate nnet_macros;

use nnet::trainer::backpropagation::IncrementalMSETrainer;
use nnet::params::{TanhNeuralNet, Default};
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, LearningRate};


ffnn!(XORNeuralNet, 2, 3, 1);


struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  #[inline(always)] fn momentum() -> f64 { 0.0f64 }
}

impl LearningRate for MyTrainerParams {
  #[inline(always)] fn lrate() -> f64 { 0.2f64 }
}


fn main() {
  let xor: [(&[f64], &[f64]); 4] = [
    (&[0f64, 0f64], &[0f64]),
    (&[0f64, 1f64], &[1f64]),
    (&[1f64, 0f64], &[1f64]),
    (&[1f64, 1f64], &[0f64])
  ];

  let mut nn: XORNeuralNet<Default> = XORNeuralNet::new();

  println!("{:?}", nn);

  {
    let mut trainer: IncrementalMSETrainer<_, _, MyTrainerParams> = 
      IncrementalMSETrainer::with_epoch_bound(&mut nn, &xor, 0.01, 9000);
    trainer.finish();
  }

  for ex in xor.iter() {
    nn.predict(ex.0);
    println!("{:?} - prediction = {:?}", ex.0, nn.output);
  }
}