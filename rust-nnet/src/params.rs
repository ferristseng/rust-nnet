use prelude::{ActivationFunction, WeightFunction, NeuralNetParameters, 
  ErrorGradient, BiasWeightFunction};

use num::Float;
use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;


/// Default Parameters for a Logistic Neural Net.
///
#[derive(Copy, Clone)] pub struct LogisticNeuralNet;

impl ActivationFunction for LogisticNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { 1f64 / (1f64 + (-x).exp()) }
  #[inline(always)] fn derivative(x: f64) -> f64 { x * (1f64 - x) }
}

impl NeuralNetParameters for LogisticNeuralNet {
  type ActivationFunction = LogisticNeuralNet;
  type WeightFunction = DefaultWeightFunction;
  type BiasWeightFunction = NegativeOneBiasFunction;
}


/// Default Parameters for a Tanh Neural Net.
///
#[derive(Copy, Clone)] pub struct TanhNeuralNet;

impl ActivationFunction for TanhNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { x.tanh() }
  #[inline(always)] fn derivative(x: f64) -> f64 { 1f64 - x.tanh().powi(2) }
}

impl NeuralNetParameters for TanhNeuralNet {
  type ActivationFunction = TanhNeuralNet;
  type WeightFunction = DefaultWeightFunction; 
  type BiasWeightFunction = PositiveOneBiasFunction;
}


#[derive(Copy, Clone)] pub struct DefaultWeightFunction;

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
///
#[derive(Copy, Clone)] pub struct DefaultErrorGradient;

impl ErrorGradient for DefaultErrorGradient {
  #[inline(always)] 
  fn errhidden<A>(act: f64, sum: f64) -> f64 where A : ActivationFunction { 
    A::derivative(act) * sum 
  }
  #[inline(always)] 
  fn erroutput<A>(exp: f64, act: f64) -> f64 where A : ActivationFunction { 
    A::derivative(act) * (exp - act) 
  }
}


/// Bias function that returns a random weight between -0.5 and 0.5.
///
#[derive(Copy, Clone)] pub struct RandomBiasWeightFunction;

impl BiasWeightFunction for RandomBiasWeightFunction {
  #[inline] 
  fn biasw() -> f64 {
    let range = Range::new(-0.5f64, 0.5f64);
    range.ind_sample(&mut thread_rng())
  }
}


/// Returns -1 for each bias node.
///
#[derive(Copy, Clone)] pub struct NegativeOneBiasFunction;

impl BiasWeightFunction for NegativeOneBiasFunction {
  #[inline] fn biasw() -> f64 { -1f64 }
}


/// Returns 1 for each bias node.
///
#[derive(Copy, Clone)] pub struct PositiveOneBiasFunction;

impl BiasWeightFunction for PositiveOneBiasFunction {
  #[inline] fn biasw() -> f64 { 1f64 }
}