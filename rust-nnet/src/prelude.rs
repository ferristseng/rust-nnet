/// Collection of parameters for a `NeuralNetTrainer`.
pub trait TrainerParameters {
  type MomentumConstant : MomentumConstant;
  type LearningRate : LearningRate;
  type ErrorGradient : ErrorGradient;
}

impl<S> TrainerParameters for S where S : MomentumConstant + LearningRate {
  type MomentumConstant = S;
  type LearningRate = S;
  type ErrorGradient = ::params::DefaultErrorGradient;
}


/// Collection of parameters for a `NeuralNet`.
pub trait NeuralNetParameters {
  type ActivationFunction : ActivationFunction;
  type WeightFunction : WeightFunction;
  type BiasWeightFunction : BiasWeightFunction;
}


/// A trainer for a single-layer neural network.
pub trait NeuralNetTrainer : Iterator { 
  #[inline] 
  fn finish(&mut self) -> Option<Self::Item> { 
    let mut current = self.next();
    
    loop {
      if current.is_none() { break; } 
      current = self.next(); 
    }

    current
  }
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
pub trait NeuralNet<P> where P : NeuralNetParameters {
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
  /// internally. The prediction can be retrieved using `layer`. 
  /// The reason `predict` doesn't return the prediction, is because it 
  /// requires a mutable borrow on `self`.
  fn predict(&mut self, inp: &[f64]);
}


// Learning Rate
pub trait LearningRate {
  fn lrate() -> f64;
}


// Momentum Constant
pub trait MomentumConstant {
  fn momentum() -> f64;
}


/// Activation Function
pub trait ActivationFunction {
  fn activation(x: f64) -> f64;
  fn derivative(x: f64) -> f64;
}


// Error Gradient method
pub trait ErrorGradient {
  fn errhidden<A>(act: f64, sum: f64) -> f64 where A : ActivationFunction;
  fn erroutput<A>(exp: f64, act: f64) -> f64 where A : ActivationFunction;
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