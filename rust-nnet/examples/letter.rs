// This example is based off of the tutorial found here:
//   
//   * https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/
//
// The basic premise is recognizing a single letter ('A') given a large dataset
// of vectors which represent pixels on a screen. The dataset is found in the 
// `data/` subfolder in this directory. It is formatted as a CSV, with the 
// first character being the expected character (one of 26 from the English 
// alphabet), and the next 16 integers form the input vector.


extern crate num;
extern crate csv;
extern crate time;
extern crate nnet;
extern crate rustc_serialize;
#[macro_use(ffnn)] extern crate nnet_macros;

use num::Float;
use time::PreciseTime;
use csv::{Reader, Result};
use nnet::trainer::backpropagation::*;
use nnet::params::{TanhNeuralNet, LogisticNeuralNet};
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, Layer, 
  LearningRate, TrainingSetMember};


// Input  = 16
// Hidden = 8
// Output = 1
ffnn!([derive(RustcEncodable, RustcDecodable)]; LetterNeuralNet, 16, 8, 1);


struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  #[inline(always)] fn momentum() -> f64 { 0.4f64 }
}

impl LearningRate for MyTrainerParams {
  #[inline(always)] fn lrate() -> f64 { 0.1f64 }
}


/// Training set example. Expected is 0 or 1. 1 if the letter is 'A', and 0 
/// otherwise.
///
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
  // Change this flag to `true`, if you want to load an already trained 
  // neural network. The training process takes a bit.
  let use_json = false;

  // Read the data and transform it into a vector of `LetterData` objects.
  let data = include_str!("data/letter-recognition.data");
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

  let mut nn: LetterNeuralNet<TanhNeuralNet> = if use_json {
    let json = include_str!("data/letter.json");
    ::rustc_serialize::json::decode(json).unwrap()
  } else {
    LetterNeuralNet::new()
  };
  
  // Use the first 2/3's of the dataset as the training set.
  let tset = 2 * rows.len() / 3; 

  println!("found {:?} examples", rows.len());
  println!("using {:?} examples to train", tset);

  if !use_json {
    let start = PreciseTime::now();
    
    parallel
      ::BatchEpochTrainer::<_, _, MyTrainerParams, _>
      ::with_epochs(&mut nn, &rows[0..tset], 500)
        .train();

    println!("took = {:?} ms", start.to(PreciseTime::now()).num_milliseconds());
  }

  let mut failed_predictions = 0;

  for (i, x) in rows.iter().enumerate() {
    nn.predict(&x.input);

    let prediction = nn.layer(Layer::Output);

    if prediction[0].round() as usize != x.expected[0].round() as usize {
      println!(
        "{:?}  | predicted = {:?} / expected = {:?}", 
        i, 
        prediction[0].round() as usize,
        x.expected[0].round() as usize);

      failed_predictions += 1;
    }
  }

  println!("failed = {:?} / total = {:?}", failed_predictions, rows.len());
}