extern crate num;

pub mod prelude;


use num::Float;
use prelude::{MLNeuralNet, ActivationFunction, MomentumConstant, LearningRate};


const DIM_IN: usize = 2;
const DIM_HI: usize = 3;
const DIM_OU: usize = 1;


pub struct ConstantLearningRate;

impl<T> LearningRate<T> for ConstantLearningRate where T : MLNeuralNet {
  fn lrate(_: &T) -> f64 { 0.1f64 }
} 


pub struct DefaultMomentumConstant;

impl MomentumConstant for DefaultMomentumConstant {
  fn momentum() -> f64 { 0.3f64 }
}


pub struct SigmoidActivationFunction;

impl ActivationFunction for SigmoidActivationFunction {
  fn activation(x: f64) -> f64 { 1f64 / (1f64 + -x.exp()) }
}


pub struct StaticMLNeuralNet {
  input   : [f64; DIM_IN + 1],
  hidden  : [f64; DIM_HI + 1],
  output  : [f64; DIM_OU],
  winput  : [[f64; DIM_HI]; DIM_IN],
  woutput : [[f64; DIM_OU]; DIM_HI]
}

impl MLNeuralNet for StaticMLNeuralNet {
  type LearningRate       = ConstantLearningRate;
  type MomentumConstant   = DefaultMomentumConstant;
  type ActivationFunction = SigmoidActivationFunction;

  #[inline(always)] fn diminput() -> usize { DIM_IN }
  #[inline(always)] fn dimhidden() -> usize { DIM_HI }
  #[inline(always)] fn dimoutput() -> usize { DIM_OU }
}

impl StaticMLNeuralNet {
  pub fn new() -> StaticMLNeuralNet {
    fn initw() -> f64 {
      0f64
    }

    StaticMLNeuralNet {
      input   : [0f64, 0f64, -1f64],
      hidden  : [0f64, 0f64, 0f64, -1f64],
      output  : [0f64],
      winput  : [[initw(), initw(), initw()], [initw(), initw(), initw()]],
      woutput : [[initw()], [initw()], [initw()]]
    }
  }
}