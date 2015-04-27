extern crate num;
extern crate rand;

mod params;
mod prelude;
mod trainer;

use std::marker::PhantomData;

pub use prelude::{NeuralNetTrainer, TrainerParameters, LearningRate, MomentumConstant,
  NNParameters, FFNeuralNet, WeightFunction};
pub use trainer::{BPTrainer};
pub use params::*;
use prelude::MutableFFNeuralNet;


const DIM_IN: usize = 2;
const DIM_HI: usize = 3;
const DIM_OU: usize = 1;


pub struct XORNeuralNet<P> {
  input   : [f64; DIM_IN + 1],
  hidden  : [f64; DIM_HI + 1],
  output  : [f64; DIM_OU],
  winput  : [[f64; DIM_HI]; DIM_IN + 1],
  woutput : [[f64; DIM_OU]; DIM_HI + 1],
  ptype   : PhantomData<P>
}

impl<P> MutableFFNeuralNet<P> for XORNeuralNet<P> where P : NNParameters {
  #[inline(always)] fn dinput(&self)  -> usize { self.input.len() - 1 }
  #[inline(always)] fn dhidden(&self) -> usize { self.hidden.len() - 1 }
  #[inline(always)] fn doutput(&self) -> usize { self.output.len() }

  #[inline(always)] fn linput(&mut self)  -> &mut [f64] { &mut self.input }
  #[inline(always)] fn lhidden(&mut self) -> &mut [f64] { &mut self.hidden }
  #[inline(always)] fn loutput(&mut self) -> &mut [f64] { &mut self.output }

  #[inline(always)] fn winhid(&mut self, i: usize) -> &mut [f64] { 
    &mut self.winput[i] 
  }
  
  #[inline(always)] fn whidou(&mut self, i: usize) -> &mut [f64] { 
    &mut self.woutput[i] 
  }
}

impl<P> XORNeuralNet<P> where P : NNParameters {
  pub fn new() -> XORNeuralNet<P> where P : NNParameters
  {
    macro_rules! initw (
      () => {
        P::WeightFunction::initw(DIM_IN, DIM_OU) 
      }
    );

    XORNeuralNet {
      input   : [0f64, 0f64, -1f64],
      hidden  : [0f64, 0f64, 0f64, -1f64],
      output  : [0f64],
      winput  : [
        [initw!(), initw!(), initw!()], 
        [initw!(), initw!(), initw!()], 
        [initw!(), initw!(), initw!()]
      ],
      woutput : [[initw!()], [initw!()], [initw!()], [initw!()]],
      ptype   : PhantomData
    }
  }
}


pub struct XORTrainingParameters;

impl<T> TrainerParameters<T> for XORTrainingParameters where T : NeuralNetTrainer {
  type MomentumConstant = DefaultMomentumConstant;
  type LearningRate     = ConstantLearningRate;
  type ErrorGradient    = DefaultErrorGradient;
}


#[test]
fn weights() {
  let xor = [
    (vec![0f64, 0f64], [0f64]),
    (vec![0f64, 1f64], [1f64]),
    (vec![1f64, 0f64], [1f64]),
    (vec![1f64, 1f64], [0f64])
  ];
  let mut nn: XORNeuralNet<TanhNeuralNet> = XORNeuralNet::new();
  let mut tr: BPTrainer<XORTrainingParameters> = BPTrainer::new(&nn);
  
  println!("");

  for _ in (0..500) {
    for &(ref v, ref e) in xor.iter() {
      nn.feedforward(&v[..]);
      tr.train(&mut nn, &e[..]);

      println!("input   : {:?}", nn.input);
      println!("weights : {:?}", nn.winput);
      println!("hidden  : {:?}", nn.hidden);
      println!("weights : {:?}", nn.woutput);
      println!("output  : {:?}", nn.output);
      println!("");
    }
  }

  assert!(false);
}
