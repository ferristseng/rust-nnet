/// Collection of parameters for a `NeuralNetTrainer` of type `T`.
pub trait TrainerParameters<T> where T : NeuralNetTrainer {
  type MomentumConstant   : MomentumConstant;
  type LearningRate       : LearningRate<T>;
  type ErrorGradient      : ErrorGradient;
}


/// Collection of parameters for a `NeuralNet`.
pub trait NNParameters {
  type ActivationFunction : ActivationFunction;
  type WeightFunction     : WeightFunction;
  type BiasWeightFunction : BiasWeightFunction;
}


pub trait NeuralNetTrainer {
  fn train<N, P0>(&mut self, nn: &mut N, ex: &[f64]) 
    where N : MutableFFNeuralNet<P0>;
}


pub trait MutableFFNeuralNet<P> {
  fn dinput(&self)  -> usize;
  fn dhidden(&self) -> usize;
  fn doutput(&self) -> usize;

  fn linput(&mut self)  -> &mut [f64];
  fn lhidden(&mut self) -> &mut [f64];
  fn loutput(&mut self) -> &mut [f64];

  fn winhid(&mut self, i: usize) -> &mut [f64];
  fn whidou(&mut self, i: usize) -> &mut [f64];
}


pub trait FFNeuralNet<P> where P : NNParameters {
  fn feedforward(&mut self, ins: &[f64]);
}

impl<T, P> FFNeuralNet<P> for T 
  where T : MutableFFNeuralNet<P>,
        P : NNParameters 
{
  fn feedforward(&mut self, ins: &[f64]) {
    assert!(ins.len() == self.dinput());

    for i in (0..self.dinput()) {
      self.linput()[i] = ins[i];
    }

    for i in (0..self.dhidden()) {
      self.lhidden()[i] = 0f64;

      for j in (0..self.dinput() + 1) {
        self.lhidden()[i] += self.linput()[j] * self.winhid(j)[i];
      }

      self.lhidden()[i] = 
        P::ActivationFunction::activation(self.lhidden()[i]);
    }

    for i in (0..self.doutput()) {
      self.loutput()[i] = 0f64;

      for j in (0..self.dhidden() + 1) {
        self.loutput()[i] += self.lhidden()[j] * self.whidou(j)[i];
      }

      self.loutput()[i] = 
        P::ActivationFunction::activation(self.loutput()[i]);
    }
  }
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


pub trait BiasWeightFunction {
  fn biasw() -> f64;
}