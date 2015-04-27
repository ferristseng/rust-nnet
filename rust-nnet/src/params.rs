use prelude::{ActivationFunction, WeightFunction, NNParameters,
  LearningRate, ErrorGradient, MomentumConstant};

use num::Float;
use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;


pub struct SigmoidNeuralNet;

impl ActivationFunction for SigmoidNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { 1f64 / (1f64 + -x.exp()) }
}

impl WeightFunction for SigmoidNeuralNet {
  #[inline] fn initw(ins: usize, outs: usize) -> f64 {
    let lb = -4f64 * (6f64 / (ins as f64 + outs as f64)).sqrt();
    let ub =  4f64 * (6f64 / (ins as f64 + outs as f64)).sqrt();
    let range = Range::new(lb, ub);
    let mut rng = thread_rng();

    range.ind_sample(&mut rng)
  }
}

impl NNParameters for SigmoidNeuralNet 
{
  type ActivationFunction = SigmoidNeuralNet;
  type WeightFunction     = SigmoidNeuralNet;
}


pub struct TanhNeuralNet;

impl ActivationFunction for TanhNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { x.tanh() }
}

impl WeightFunction for TanhNeuralNet {
  #[inline] fn initw(ins: usize, outs: usize) -> f64 {
    let lb = -(6f64 / (ins as f64 + outs as f64)).sqrt();
    let ub =  (6f64 / (ins as f64 + outs as f64)).sqrt();
    let range = Range::new(lb, ub);
    let mut rng = thread_rng();

    range.ind_sample(&mut rng)
  }
}

impl NNParameters for TanhNeuralNet {
  type ActivationFunction = TanhNeuralNet;
  type WeightFunction     = TanhNeuralNet; 
}


pub struct ConstantLearningRate;

impl<T> LearningRate<T> for ConstantLearningRate {
  #[inline(always)] fn lrate(_: &T) -> f64 { 0.7f64 }
} 


pub struct DefaultMomentumConstant;

impl MomentumConstant for DefaultMomentumConstant {
  #[inline(always)] fn momentum() -> f64 { 0.9f64 }
}


pub struct DefaultErrorGradient;

impl ErrorGradient for DefaultErrorGradient {
  fn errhidden(act: f64, sum: f64) -> f64 { act * (1f64 - act) * sum }
  fn erroutput(exp: f64, act: f64) -> f64 { act * (1f64 - act) * (act - exp) }
}