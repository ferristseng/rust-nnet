extern crate nnet;
extern crate rustc_serialize;
#[macro_use(ffnn)] extern crate nnet_macros;


use rustc_serialize::json;
use nnet::params::{TanhNeuralNet, LogisticNeuralNet};

/// The `ffnn!` macro can take in any number of meta arguments as its first 
/// parameter. For this structure, we are automatically deriving 
/// `RustcEncodable` and `RustcDecodable` from the `rustc_serialize` crate.
ffnn!([derive(RustcDecodable, RustcEncodable)]; XORNeuralNetDerived, 2, 3, 1);


/// Since the `ffnn!` macro defines the structure in this crate. We can 
/// define a custom way to serialize, and deserialize the neural network.
ffnn!(XORNeuralNetCustomImpl, 2, 3, 1);


fn main() {
  let decoded: XORNeuralNetDerived<TanhNeuralNet> = 
    json::decode(include_str!("data/xor.json")).unwrap();

  println!("{}", json::as_pretty_json(&decoded));
}