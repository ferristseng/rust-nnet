extern crate num;
extern crate rand;

pub mod prelude;


use std::marker::PhantomData;


use num::Float;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use prelude::*;


const DIM_IN: usize = 2;
const DIM_HI: usize = 3;
const DIM_OU: usize = 1;


pub struct ConstantLearningRate;

impl<T> LearningRate<T> for ConstantLearningRate {
  #[inline(always)] fn lrate(_: &T) -> f64 { 0.1f64 }
} 


pub struct DefaultMomentumConstant;

impl MomentumConstant for DefaultMomentumConstant {
  #[inline(always)] fn momentum() -> f64 { 0.3f64 }
}


pub struct DefaultErrorGradient;

impl ErrorGradient for DefaultErrorGradient {
  fn errhidden(act: f64, sum: f64) -> f64 { act * (1f64 - act) * sum }
  fn erroutput(exp: f64, act: f64) -> f64 { act * (1f64 - act) * (act - exp) }
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
    let mut rng = rand::thread_rng();

    range.ind_sample(&mut rng)
  }
}

impl<T> Parameters<T> for TanhNeuralNet where T : MLNeuralNet {
  type LearningRate       = ConstantLearningRate;
  type ErrorGradient      = DefaultErrorGradient;
  type MomentumConstant   = DefaultMomentumConstant;
  type ActivationFunction = TanhNeuralNet;
  type WeightFunction     = TanhNeuralNet; 
}


pub struct SigmoidNeuralNet;

impl ActivationFunction for SigmoidNeuralNet {
  #[inline(always)] fn activation(x: f64) -> f64 { 1f64 / (1f64 + -x.exp()) }
}

impl WeightFunction for SigmoidNeuralNet {
  #[inline] fn initw(ins: usize, outs: usize) -> f64 {
    let lb = -4f64 * (6f64 / (ins as f64 + outs as f64)).sqrt();
    let ub =  4f64 * (6f64 / (ins as f64 + outs as f64)).sqrt();
    let range = Range::new(lb, ub);
    let mut rng = rand::thread_rng();

    range.ind_sample(&mut rng)
  }
}

impl<T> Parameters<T> for SigmoidNeuralNet where T : MLNeuralNet 
{
  type LearningRate       = ConstantLearningRate;
  type ErrorGradient      = DefaultErrorGradient;
  type MomentumConstant   = DefaultMomentumConstant;
  type ActivationFunction = SigmoidNeuralNet;
  type WeightFunction     = SigmoidNeuralNet;
}


pub struct XORNeuralNet<P>
{
  input   : [f64; DIM_IN + 1],
  hidden  : [f64; DIM_HI + 1],
  output  : [f64; DIM_OU],
  winput  : [[f64; DIM_HI]; DIM_IN + 1],
  woutput : [[f64; DIM_OU]; DIM_HI + 1],
  dinput  : [[f64; DIM_HI]; DIM_IN + 1],
  doutput : [[f64; DIM_OU]; DIM_HI + 1], 
  ehidden : [f64; DIM_HI + 1],
  eoutput : [f64; DIM_OU + 1],
  ptype   : PhantomData<P>
}

impl<P> MLNeuralNet for XORNeuralNet<P> where P : Parameters<XORNeuralNet<P>> 
{
  #[inline(always)] fn diminput() -> usize { DIM_IN }
  #[inline(always)] fn dimhidden() -> usize { DIM_HI }
  #[inline(always)] fn dimoutput() -> usize { DIM_OU }

  fn feedforward(&mut self, ins: &[f64]) {
    assert!(ins.len() == Self::diminput());

    for i in (0..Self::diminput()) {
      self.input[i] = ins[i];
    }

    for i in (0..Self::dimhidden()) {
      self.hidden[i] = 0f64;

      for j in (0..Self::diminput() + 1) {
        self.hidden[i] += self.input[j] * self.winput[j][i];
      }

      self.hidden[i] = P::ActivationFunction::activation(self.hidden[i]);
    }

    for i in (0..Self::dimoutput()) {
      self.output[i] = 0f64;

      for j in (0..Self::dimhidden() + 1) {
        self.output[i] += self.hidden[j] * self.woutput[j][i];
      }

      self.output[i] = P::ActivationFunction::activation(self.output[i]);
    }
  }

  fn backpropagate(&mut self, exp: &[f64]) {
    assert!(exp.len() == Self::dimoutput());

    for i in (0..Self::dimoutput()) {
      self.eoutput[i] = P::ErrorGradient::erroutput(exp[i], self.output[0]);

      for j in (0..Self::dimhidden() + 1) {
        self.doutput[j][i] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
        self.hidden[j] * self.eoutput[i] + 
        P::MomentumConstant::momentum() * self.doutput[j][i];
      }
    }

    for i in (0..Self::dimhidden()) {
      let wsum = (0..Self::dimoutput())
        .fold(0f64, |acc, j| acc + (self.woutput[i][j] * self.eoutput[j]));

      self.ehidden[i] = P::ErrorGradient::errhidden(self.hidden[i], wsum);

      for j in (0..Self::diminput() + 1) {
        self.dinput[j][i] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
        self.input[j] * self.ehidden[i] + 
        P::MomentumConstant::momentum() * self.dinput[j][i];
      }
    }

    for i in (0..Self::diminput() + 1) {
      for j in (0..Self::dimhidden()) {
        self.winput[i][j] += self.dinput[i][j];
      }
    }

    for i in (0..Self::dimhidden() + 1) {
      for j in (0..Self::dimoutput()) {
        self.woutput[i][j] += self.doutput[i][j];
      }
    }
  }
}

impl<P> XORNeuralNet<P> where P : Parameters<XORNeuralNet<P>>  
{
  pub fn new() -> XORNeuralNet<P> where P : Parameters<XORNeuralNet<P>>
  {
    macro_rules! initw (
      () => {
        P::WeightFunction::initw(Self::diminput(), Self::dimoutput()) 
      }
    );

    XORNeuralNet {
      input   : [0f64, 0f64, -1f64],
      hidden  : [0f64, 0f64, 0f64, -1f64],
      output  : [0f64],
      winput  : [
        [initw!(), initw!(), initw!()], 
        [initw!(), initw!(), initw!()], 
        [initw!(), initw!(), initw!()]
      ],
      woutput : [[initw!()], [initw!()], [initw!()], [initw!()]],
      dinput  : [[0f64, 0f64, 0f64], [0f64, 0f64, 0f64], [0f64, 0f64, 0f64]],
      doutput : [[0f64], [0f64], [0f64], [0f64]],
      ehidden : [0f64, 0f64, 0f64, 0f64],
      eoutput : [0f64, 0f64],
      ptype   : PhantomData
    }
  }
}


#[test]
fn weights() {
  let xor = [
    (vec![0f64, 0f64], [0f64]),
    (vec![0f64, 1f64], [1f64]),
    (vec![1f64, 0f64], [1f64]),
    (vec![1f64, 1f64], [0f64])
  ];
  let mut nn: XORNeuralNet<SigmoidNeuralNet> = XORNeuralNet::new();
  
  println!("");

  for &(ref v, ref e) in xor.iter() {
    nn.feedforward(&v[..]);
    nn.backpropagate(&e[..]);

    println!("input   : {:?}", nn.input);
    println!("hidden  : {:?}", nn.hidden);
    println!("output  : {:?}", nn.output);
    println!("");
  }

  assert!(false);
}
