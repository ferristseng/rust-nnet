extern crate num;
extern crate rand;

pub mod params;
pub mod prelude;
pub mod trainer;

use std::marker::PhantomData;
use std::fmt::{Debug, Error, Formatter};

#[cfg(test)]
use trainer::backpropagation::{SequentialEpochTrainer};

pub use prelude::*;
pub use params::*;

use prelude::WeightLayer::*;


const DIM_IN: usize = 2;
const DIM_HI: usize = 2;
const DIM_OU: usize = 1;


pub struct XORNeuralNet<P> {
  input   : [f64; DIM_IN + 1],
  hidden  : [f64; DIM_HI + 1],
  output  : [f64; DIM_OU],
  winput  : [[f64; DIM_HI]; DIM_IN + 1],
  woutput : [[f64; DIM_OU]; DIM_HI + 1],
  ptype   : PhantomData<P>
}

impl<P> NeuralNet for XORNeuralNet<P> where P : NNParameters {
  #[inline(always)] fn dim_input() -> usize { DIM_IN }

  #[inline(always)] fn dim_output() -> usize { DIM_OU }

  #[inline(always)] fn dim_hidden() -> usize { DIM_HI }

  #[inline] fn update_weight(&mut self, layer: WeightLayer, w: f64) {
    match layer {
      InputHidden(x, y) => self.winput[x][y] = w,
      HiddenOutput(x, y) => self.woutput[x][y] = w
    }
  }

  #[inline] fn weight(&self, layer: WeightLayer) -> f64 {
    match layer {
      InputHidden(x, y) => self.winput[x][y],
      HiddenOutput(x, y) => self.woutput[x][y]
    }
  }

  #[inline(always)] fn hidden_node(&self, i: usize) -> f64 { self.hidden[i] }

  #[inline(always)] fn output_layer(&self) -> &[f64] { self.output.as_ref() }

  #[inline(always)] fn input_layer(&self) -> &[f64] { self.input.as_ref() }

  fn predict(&mut self, inp: &[f64]) {
    assert!(inp.len() == Self::dim_input());

    for i in (0..Self::dim_input()) {
      self.input[i] = inp[i];
    }

    for i in (0..Self::dim_hidden()) {
      self.hidden[i] = 0f64;

      for j in (0..Self::dim_input() + 1) {
        self.hidden[i] += self.input[j] * self.weight(InputHidden(j, i));
      }

      self.hidden[i] = 
        P::ActivationFunction::activation(self.hidden_node(i));
    }

    for i in (0..Self::dim_output()) {
      self.output[i] = 0f64;

      for j in (0..Self::dim_hidden() + 1) {
        self.output[i] += self.hidden[j] * self.weight(HiddenOutput(j, i));
      }

      self.output[i] = 
        P::ActivationFunction::activation(self.output[i]);
    }
  }
}

impl<P> XORNeuralNet<P> where P : NNParameters {
  pub fn new() -> XORNeuralNet<P> where P : NNParameters {
    macro_rules! initw (
      () => {
        P::WeightFunction::initw(DIM_IN, DIM_OU) 
      }
    );

    macro_rules! biasw (
      () => {
        P::BiasWeightFunction::biasw()
      }
    );

    XORNeuralNet {
      input   : [0f64, 0f64, biasw!()],
      hidden  : [0f64, 0f64, biasw!()],
      output  : [0f64],
      winput  : [
        [initw!(), initw!()],
        [initw!(), initw!()],
        [initw!(), initw!()] 
      ],
      woutput : [[-0.993423f64], [0.164732f64], [0.752621f64]],
      ptype   : PhantomData
    }
  }
}


struct XORTrainingParameters;

impl<T> LearningRate<T> for XORTrainingParameters {
  #[inline(always)] fn lrate(_: &T) -> f64 { 0.3f64 }
} 

impl MomentumConstant for XORTrainingParameters {
  #[inline(always)] fn momentum() -> f64 { 0.8f64 }
}

impl<T> TrainerParameters<T> for XORTrainingParameters 
  where T : NeuralNetTrainer 
{
  type MomentumConstant = XORTrainingParameters;
  type LearningRate     = XORTrainingParameters;
  type ErrorGradient    = DefaultErrorGradient;
}

impl<P> Debug for XORNeuralNet<P> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    write!(
      f, 
      "---\ninput {:?}\nweights {:?}\nhidden {:?}\nweights {:?}\noutput {:?}\n---\n", 
      self.input,
      self.winput,
      self.hidden,
      self.woutput,
      self.output)
  }
}


#[test]
fn weights() {
  println!("");

  let xor: [(&[f64], &[f64]); 4] = [
    (&[0f64, 0f64], &[0f64]),
    (&[0f64, 1f64], &[1f64]),
    (&[1f64, 0f64], &[1f64]),
    (&[1f64, 1f64], &[0f64])
  ];
  let mut nn: XORNeuralNet<SigmoidNeuralNet> = XORNeuralNet::new();
  let tr: SequentialEpochTrainer<XORTrainingParameters> = 
    SequentialEpochTrainer::new(5000);
  
  tr.train(&mut nn, &xor);

  for ex in xor.iter() {
    nn.predict(ex.0);
    println!("{:?} - prediction = {:?}", ex.0, nn.output_layer());
  }

  assert!(false);
}
