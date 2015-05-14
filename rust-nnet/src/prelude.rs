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


/// A trainer for a single-layer neural network.
pub trait NeuralNetTrainer {
  fn train<N, T>(&self, nn: &mut N, ex: &[T]) 
    where N : NeuralNet + ::std::fmt::Debug, T : TrainingSetMember;
}


/// Layer names.
pub enum Layer {
  Input,
  Hidden,
  Output
}


/// Coordinates for a node in a specified layer.
pub enum Node {
  Input(usize),
  Hidden(usize),
  Output(usize),
  WeightInputHidden(usize, usize),
  WeightHiddenOutput(usize, usize)
}


/// A single-layer neural network.
pub trait NeuralNet {
  /// Returns the dimensions of the input layer.
  fn dim_input() -> usize;

  /// Returns the dimensions of the output layer.
  fn dim_output() -> usize;

  /// Returns the dimensions of the hidden layer.
  fn dim_hidden() -> usize;

  /// Returns the value of a node in a layer at a specified coordinate.
  fn node(&self, i: Node) -> f64;

  /// Returns a mutable reference to a node in a layer.
  fn node_mut(&mut self, i: Node) -> &mut f64;

  /// Returns the specified layer.
  fn layer(&self, layer: Layer) -> &[f64];

  /// Computes the predicted value for a given input and stores it 
  /// internally. The prediction can be retrieved using `output_layer`. 
  /// The reason `predict` doesn't return the prediction, is because it 
  /// requires a mutable borrow on `self`.
  fn predict(&mut self, inp: &[f64]);
}


// Learning Rate
pub trait LearningRate<T> {
  fn lrate(nn: &T) -> f64;
}


// Momentum Constant
pub trait MomentumConstant {
  fn momentum() -> f64;
}


///
pub trait ActivationFunction {
  fn activation(x: f64) -> f64;
}


// Error Gradient
pub trait ErrorGradient {
  fn errhidden(act: f64, sum: f64) -> f64;
  fn erroutput(exp: f64, act: f64) -> f64;
}


/// The weight function to generate the initial weights. 
pub trait WeightFunction {
  fn initw(ins: usize, outs: usize) -> f64;
}


/// The weight function to generate the bias nodes' weights.
pub trait BiasWeightFunction {
  fn biasw() -> f64;
}


/// A member of a training set.
pub trait TrainingSetMember {
  fn expected(&self) -> &[f64];
  fn input(&self) -> &[f64];
}

impl<'a> TrainingSetMember for (&'a [f64], &'a [f64]) {
  fn expected(&self) -> &[f64] { self.1 }
  fn input(&self) -> &[f64] { self.0 }
}