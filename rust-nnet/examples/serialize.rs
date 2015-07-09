extern crate nnet;
extern crate rustc_serialize;
#[macro_use(ffnn)] extern crate nnet_macros;

use rustc_serialize::json;
use rustc_serialize::{Decodable, Decoder};
use nnet::params::{TanhNeuralNet, LogisticNeuralNet};
use nnet::prelude::{NeuralNet, Layer};

/// The `ffnn!` macro can take in any number of meta arguments as its first 
/// parameter. For this structure, we are automatically deriving 
/// `RustcEncodable` and `RustcDecodable` from the `rustc_serialize` crate.
ffnn!([derive(RustcDecodable, RustcEncodable)]; XORNeuralNetDerived, 2, 3, 1);


/// Since the `ffnn!` macro defines the structure in this crate. We can 
/// define a custom way to serialize, and deserialize the neural network.
ffnn!(XORNeuralNetCustomImpl, 2, 3, 1);

impl<P> Decodable for XORNeuralNetCustomImpl<P> {
  fn decode<D : Decoder>(d: &mut D) -> Result<Self, D::Error> {
    // Not going to implement this, but, because XORNeuralNetCustomImpl is 
    // defined by the macro in this module, it is possible to implement traits 
    // for it.
    Err(d.error("not implemented"))
  }
}

fn main() {
  let xor: [(&[f64], &[f64]); 4] = [
    (&[0f64, 0f64], &[0f64]),
    (&[0f64, 1f64], &[1f64]),
    (&[1f64, 0f64], &[1f64]),
    (&[1f64, 1f64], &[0f64])
  ];

  let mut decoded: XORNeuralNetDerived<TanhNeuralNet> = 
    json::decode(include_str!("data/xor.json")).unwrap();

  for ex in xor.iter() {
    decoded.predict(ex.0);
    println!("{:?} - prediction = {:?}", ex.0, decoded.layer(Layer::Output));
  }
}
