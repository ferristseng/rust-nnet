use num;

use prelude::*;


/// State of a trainer. Used internally.
pub struct TrainerState {
  dinput  : Vec<Vec<f64>>,
  doutput : Vec<Vec<f64>>, 
  ehidden : Vec<f64>,
  eoutput : Vec<f64>,
} 

impl TrainerState {
  pub fn new<N>(_: &N) -> TrainerState where N : NeuralNet {
    let mut state = TrainerState {
      dinput  : Vec::with_capacity(N::dim_input() + 1),
      doutput : Vec::with_capacity(N::dim_hidden() + 1),
      ehidden : Vec::with_capacity(N::dim_hidden() + 1),
      eoutput : Vec::with_capacity(N::dim_output() + 1)
    };

    for _ in (0..N::dim_input() + 1) {
      let _v: Vec<f64> = (0..N::dim_hidden()).map(|_| 0f64).collect();
      state.dinput.push(_v);
    }

    for _ in (0..N::dim_hidden() + 1) {
      let _v: Vec<f64> = (0..N::dim_output()).map(|_| 0f64).collect(); 
      state.doutput.push(_v);
      state.ehidden.push(0f64);
    }

    for _ in (0..N::dim_output() + 1) { state.eoutput.push(0f64); }

    state
  }
}


/// Compares a neural network's prediction for an input, and calculates the 
/// error given an expected result. Updates the state with the deltas and errors 
/// of the hidden and output layers.
pub fn update_state<P, T, N, M>(t: &T, nn: &mut N, state: &mut TrainerState, member: &M)
  where P : TrainerParameters<T>,
        T : NeuralNetTrainer,
        N : NeuralNet, 
        M : TrainingSetMember,
{
  let exp = member.expected();
  
  nn.predict(member.input());

  let res = nn.layer(Layer::Output);
  let inp = nn.layer(Layer::Input);

  for i in (0..N::dim_output()) {
    state.eoutput[i] = P::ErrorGradient::erroutput(exp[i], res[i]);

    for j in (0..N::dim_hidden() + 1) {
      state.doutput[j][i] = P::LearningRate::lrate(t) * 
        nn.node(Node::Hidden(j)) * state.eoutput[i] + 
        P::MomentumConstant::momentum() * state.doutput[j][i];
    }
  }

  for i in (0..N::dim_hidden()) {
    let wsum = (0..N::dim_output()).fold(
      0f64, 
      |acc, j| acc + (nn.node(Node::WeightHiddenOutput(i, j)) * state.eoutput[j]));

    state.ehidden[i] = P::ErrorGradient::errhidden(nn.node(Node::Hidden(i)), wsum);

    for j in (0..N::dim_input() + 1) {
      state.dinput[j][i] = P::LearningRate::lrate(t) * 
        inp[j] * state.ehidden[i] + 
        P::MomentumConstant::momentum() * state.dinput[j][i];
    }
  }
}


/// Update weights in each layer of a neural network with a single hidden layer.
pub fn update_weights<N>(nn: &mut N, state: &TrainerState) where N : NeuralNet {
  for i in (0..N::dim_input() + 1) {
    for j in (0..N::dim_hidden()) {
      let w = nn.node(Node::WeightInputHidden(i, j));
      *nn.node_mut(Node::WeightInputHidden(i, j)) = w + state.dinput[i][j];
    }
  }

  for i in (0..N::dim_hidden() + 1) {
    for j in (0..N::dim_output()) {
      let w = nn.node(Node::WeightHiddenOutput(i, j));
      *nn.node_mut(Node::WeightHiddenOutput(i, j)) = w + state.doutput[i][j];
    }
  }
}


/// Calculates the Mean-Squared-Error from a vector of predictions, and expected 
/// values. 
pub fn mse<'a, I>(predictions: I, expected: I) -> f64 where I : Iterator<Item=&'a f64> {
  let mut n = 0f64;
  let sum = predictions
    .zip(expected)
    .fold(0f64, |acc, (act, exp)| { n += 1f64; acc + num::pow((act - exp), 2) });
  (1f64 / n) * sum
}