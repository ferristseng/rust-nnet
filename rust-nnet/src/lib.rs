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

    self.input[0] = ins[0];
    self.input[1] = ins[1];

    self.hidden[0] = 0f64;
    self.hidden[1] = 0f64;
    self.hidden[2] = 0f64;

    self.hidden[0] += self.input[0] * self.winput[0][0];
    self.hidden[0] += self.input[1] * self.winput[1][0];
    self.hidden[0] += self.input[2] * self.winput[2][0];

    self.hidden[0] = P::ActivationFunction::activation(self.hidden[0]);

    self.hidden[1] += self.input[0] * self.winput[0][1];
    self.hidden[1] += self.input[1] * self.winput[1][1];
    self.hidden[1] += self.input[2] * self.winput[2][1];

    self.hidden[1] = P::ActivationFunction::activation(self.hidden[1]);

    self.hidden[2] += self.input[0] * self.winput[0][2];
    self.hidden[2] += self.input[1] * self.winput[1][2];
    self.hidden[2] += self.input[2] * self.winput[2][2];

    self.hidden[2] = P::ActivationFunction::activation(self.hidden[2]);

    self.output[0] = 0f64;

    self.output[0] += self.hidden[0] * self.woutput[0][0];
    self.output[0] += self.hidden[1] * self.woutput[1][0];
    self.output[0] += self.hidden[2] * self.woutput[2][0];
    self.output[0] += self.hidden[3] * self.woutput[3][0];

    self.output[0] = P::ActivationFunction::activation(self.output[0]);
  }

  fn backpropagate(&mut self, exp: &[f64]) {
    assert!(exp.len() == Self::dimoutput());

    self.eoutput[0] = P::ErrorGradient::erroutput(exp[0], self.output[0]); 

    self.doutput[0][0] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
      self.hidden[0] * self.eoutput[0] + 
      P::MomentumConstant::momentum() * self.doutput[0][0];
    self.doutput[1][0] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
      self.hidden[1] * self.eoutput[0] + 
      P::MomentumConstant::momentum() * self.doutput[1][0];
    self.doutput[2][0] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
      self.hidden[2] * self.eoutput[0] + 
      P::MomentumConstant::momentum() * self.doutput[2][0];
    self.doutput[3][0] = P::LearningRate::lrate(&self as &XORNeuralNet<P>) * 
      self.hidden[3] * self.eoutput[0] + 
      P::MomentumConstant::momentum() * self.doutput[3][0];
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
