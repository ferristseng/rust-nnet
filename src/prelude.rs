pub trait MLNeuralNet {
  type MomentumConstant   : MomentumConstant;
  type LearningRate       : LearningRate<Self>;
  type ActivationFunction : ActivationFunction;

  fn diminput() -> usize;
  fn dimhidden() -> usize;
  fn dimoutput() -> usize;
}


// α - Learning Rate
pub trait LearningRate<T> where T : MLNeuralNet {
  fn lrate(nn: &T) -> f64;
}


// β - Momentum Constant
pub trait MomentumConstant {
  fn momentum() -> f64;
}


pub trait ActivationFunction {
  fn activation(x: f64) -> f64;
}