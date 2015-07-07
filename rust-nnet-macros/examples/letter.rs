// This example is based off of the tutorial found here:
//   
//   * https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/
//
// The basic premise is recognizing a single letter ('A') given a large dataset
// of vectors which represent pixels on a screen. The dataset is found in the 
// `data/` subfolder in this directory. It is formatted as a CSV, with the 
// first character being the expected character (one of 26 from the English 
// alphabet), and the next 16 integers form the input vector.


extern crate csv;
extern crate time;
extern crate nnet;
extern crate rustc_serialize;
#[macro_use(ffnn)] extern crate nnet_macros;

use csv::{Reader, Result};
use time::PreciseTime;
use nnet::trainer::backpropagation::*;
use nnet::params::{TanhNeuralNet, LogisticNeuralNet};
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, 
  Layer, LearningRate, TrainingSetMember};


// Input  = 16
// Hidden = 8
// Output = 1
ffnn!(LetterNeuralNet, 16, 8, 1);


struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  #[inline(always)] fn momentum() -> f64 { 0.4f64 }
}

impl LearningRate for MyTrainerParams {
  #[inline(always)] fn lrate() -> f64 { 0.1f64 }
}


// Expected is 1 or 0. 1 if the expected letter is 'A',
// and 0 otherwise.
#[derive(RustcDecodable)]
struct LetterData {
  expected: [f64; 1],
  input: [f64; 16]
}

impl TrainingSetMember for LetterData {
  #[inline(always)] fn input(&self) -> &[f64] { self.input.as_ref() }
  #[inline(always)] fn expected(&self) -> &[f64] { self.expected.as_ref() }
}


fn main() {
  let data = include_str!("data/letter-recognition.data");

  // Read the data and transform it into a vector of `LetterData` objects.
  let rows = Reader::from_string(data)
    .has_headers(false)
    .decode()
    .map(|decoded: Result<(char, _)>| {
      match decoded {
        Ok((c, input)) => {
          let is_a = if c == 'A' { 1f64 } else { 0f64 };
          LetterData { expected: [is_a; 1], input: input }
        }
        Err(e) => panic!("unrecognzed data: {:?}", e)
      }
    })
    .collect::<Vec<LetterData>>();

  let mut nn: LetterNeuralNet<TanhNeuralNet> = LetterNeuralNet::new();
  let start = PreciseTime::now();
  
  // Use the first 2/3's of the dataset as the training set.
  let tset = 2 * rows.len() / 3; 

  println!("found {:?} examples", rows.len());
  println!("training set = 0..{:?}", tset);

  BatchEpochTrainer::<_, _, MyTrainerParams, _>::new(&mut nn, &rows[0..tset], 500).finish();

  println!("took = {:?} ms", start.to(PreciseTime::now()).num_milliseconds());
}