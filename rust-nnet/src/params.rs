use prelude::{ActivationFunction, WeightFunction, NNParameters, 
  ErrorGradient, BiasWeightFunction};

use num::Float;
use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;


/// Default Parameters for a Logistic Neural Net.
pub struct LogisticNeuralNet;

impl ActivationFunction for LogisticNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { 1f64 / (1f64 + (-x).exp()) }
}

impl NNParameters for LogisticNeuralNet {
  type ActivationFunction = LogisticNeuralNet;
  type WeightFunction = DefaultWeightFunction;
  type BiasWeightFunction = NegativeOneBiasFunction;
}


/// Default Parameters for a Tanh Neural Net.
pub struct TanhNeuralNet;

impl ActivationFunction for TanhNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { x.tanh() }
}

impl NNParameters for TanhNeuralNet {
  type ActivationFunction = TanhNeuralNet;
  type WeightFunction = DefaultWeightFunction; 
  type BiasWeightFunction = PositiveOneBiasFunction;
}


pub struct DefaultWeightFunction;

impl WeightFunction for DefaultWeightFunction {
  #[inline] 
  fn initw(ins: usize, _: usize) -> f64 {
    let lb = -1f64 / (ins as f64).sqrt();
    let ub =  1f64 / (ins as f64).sqrt();
    let range = Range::new(lb, ub);

    range.ind_sample(&mut thread_rng())
  }
}


/// Default Error Gradient functions.
pub struct DefaultErrorGradient;

impl ErrorGradient for DefaultErrorGradient {
  fn errhidden(act: f64, sum: f64) -> f64 { act * (1f64 - act) * sum }
  fn erroutput(exp: f64, act: f64) -> f64 { act * (1f64 - act) * (exp - act) }
}


/// Bias function that returns a random weight between -0.5 and 0.5.
pub struct RandomBiasWeightFunction;

impl BiasWeightFunction for RandomBiasWeightFunction {
  #[inline] 
  fn biasw() -> f64 {
    let range = Range::new(-0.5f64, 0.5f64);
    range.ind_sample(&mut thread_rng())
  }
}


/// Returns -1 for each bias node.
pub struct NegativeOneBiasFunction;

impl BiasWeightFunction for NegativeOneBiasFunction {
  #[inline] fn biasw() -> f64 { -1f64 }
}


/// Returns 1 for each bias node.
pub struct PositiveOneBiasFunction;

impl BiasWeightFunction for PositiveOneBiasFunction {
  #[inline] fn biasw() -> f64 { 1f64 }
}


/// Default parameters for a NeuralNet.
pub type Default = LogisticNeuralNet;