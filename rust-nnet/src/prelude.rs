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
    where N : NeuralNet, T : TrainingSetMember;
}


/// Possible places weights could be stored within a NeuralNetwork.
pub enum WeightLayer {
  InputHidden(usize, usize),
  HiddenOutput(usize, usize)
}


/// A single-layer neural network.
pub trait NeuralNet {
  /// Returns the dimensions of the input layer.
  fn dim_input() -> usize;

  /// Returns the dimensions of the output layer.
  fn dim_output() -> usize;

  /// Returns the dimensions of the hidden layer.
  fn dim_hidden() -> usize;

  /// Updates the weights for an item at the coordinates described by a 
  /// `WeightLayer`.
  fn update_weight(&mut self, layer: WeightLayer, w: f64);

  /// Returns a weight for the coordinates described by a `WeightLayer`.
  fn weight(&self, layer: WeightLayer) -> f64;

  /// Returns the value of a hidden node at the specified index.
  fn hidden_node(&self, i: usize) -> f64;

  /// Returns the output layer.
  fn output_layer(&self) -> &[f64];

  /// Returns the input layer.
  fn input_layer(&self) -> &[f64];

  /// Computes the predicted value for a given input and stores it 
  /// internally. The prediction can be retrieved using `output_layer`. 
  /// The reason `predict` doesn't return the prediction, is because it 
  /// requires a mutable borrow on `self`.
  fn predict(&mut self, inp: &[f64]);
}


// α - Learning Rate
pub trait LearningRate<T> {
  fn lrate(nn: &T) -> f64;
}


// β - Momentum Constant
pub trait MomentumConstant {
  fn momentum() -> f64;
}


///
pub trait ActivationFunction {
  fn activation(x: f64) -> f64;
}


// δ - Error Gradient
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

impl TrainingSetMember for (Vec<f64>, Vec<f64>) {
  fn expected(&self) -> &[f64] { &self.1[..] }
  fn input(&self) -> &[f64] { &self.0[..] }
}