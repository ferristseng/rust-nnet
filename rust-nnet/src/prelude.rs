// Collection of parameters for a `MLNeuralNet` of type `T`.
pub trait Parameters<T> where T : MLNeuralNet {
  type MomentumConstant   : MomentumConstant;
  type LearningRate       : LearningRate<T>;
  type ErrorGradient      : ErrorGradient;
  type ActivationFunction : ActivationFunction;
  type WeightFunction     : WeightFunction;
}

pub trait MLNeuralNet {
  fn diminput() -> usize;
  fn dimhidden() -> usize;
  fn dimoutput() -> usize;

  fn feedforward(&mut self, ins: &[f64]);
  fn backpropagate(&mut self, exp: &[f64]);
}


// α - Learning Rate
pub trait LearningRate<T> {
  fn lrate(nn: &T) -> f64;
}


// β - Momentum Constant
pub trait MomentumConstant {
  fn momentum() -> f64;
}


pub trait ActivationFunction {
  fn activation(x: f64) -> f64;
}


// δ - Error Gradient
pub trait ErrorGradient {
  fn errhidden(act: f64, sum: f64) -> f64;
  fn erroutput(exp: f64, act: f64) -> f64;
}


pub trait WeightFunction {
  fn initw(ins: usize, outs: usize) -> f64;
}