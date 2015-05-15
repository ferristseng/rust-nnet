#![feature(plugin)]
#![plugin(nnet_macros)]

extern crate nnet;

use nnet::params::LogisticNeuralNet;

create_ffnn!(XORNeuralNet, 2, 3, 1);

fn main() {
  let nn: XORNeuralNet<LogisticNeuralNet> = XORNeuralNet::new();
}