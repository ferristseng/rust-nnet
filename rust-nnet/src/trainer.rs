use std::marker::PhantomData;

use prelude::*;


pub struct BPTrainer<P> {
  dinput  : Vec<Vec<f64>>,
  doutput : Vec<Vec<f64>>, 
  ehidden : Vec<f64>,
  eoutput : Vec<f64>,
  ptype   : PhantomData<P>
}

impl<P> NeuralNetTrainer for BPTrainer<P>
  where P : TrainerParameters<BPTrainer<P>>
{
  fn train<N, P0>(&mut self, nn: &mut N, exp: &[f64]) where N : MutableFFNeuralNet<P0> {
    assert!(exp.len() == nn.doutput());

    for i in (0..nn.doutput()) {
      self.eoutput[i] = P::ErrorGradient::erroutput(exp[i], nn.loutput()[i]);

      println!("erroutput: {:?} (exp: {:?}, act: {:?})", self.eoutput[i], exp[i], nn.loutput()[i]);

      for j in (0..nn.dhidden() + 1) {
        self.doutput[j][i] = P::LearningRate::lrate(&self) * 
          nn.lhidden()[j] * self.eoutput[i] + 
          P::MomentumConstant::momentum() * self.doutput[j][i];
      }
    }

    for i in (0..nn.dhidden()) {
      let wsum = (0..nn.doutput())
        .fold(0f64, |acc, j| acc + (nn.whidou(i)[j] * self.eoutput[j]));

      println!("{:?}", wsum);

      self.ehidden[i] = P::ErrorGradient::errhidden(nn.lhidden()[i], wsum);

      for j in (0..nn.dinput() + 1) {
        self.dinput[j][i] = P::LearningRate::lrate(&self) * 
          nn.linput()[j] * self.ehidden[i] + 
          P::MomentumConstant::momentum() * self.dinput[j][i];
      }
    }

    for i in (0..nn.dinput() + 1) {
      for j in (0..nn.dhidden()) {
        nn.winhid(i)[j] += self.dinput[i][j];
      }
    }

    for i in (0..nn.dhidden() + 1) {
      for j in (0..nn.doutput()) {
        nn.whidou(i)[j] += self.doutput[i][j];
      }
    }
  }
}

impl<P> BPTrainer<P> 
  where P : TrainerParameters<BPTrainer<P>>
{
  pub fn new<N, P0>(nn: &N) -> BPTrainer<P> 
    where N : MutableFFNeuralNet<P0>
  {
    let mut trainer = BPTrainer {
      dinput  : Vec::with_capacity(nn.dinput() + 1),
      doutput : Vec::with_capacity(nn.dhidden() + 1),
      ehidden : Vec::with_capacity(nn.dhidden() + 1),
      eoutput : Vec::with_capacity(nn.doutput() + 1),
      ptype   : PhantomData
    };

    for _ in (0..nn.dinput() + 1) {
      let mut _v = Vec::with_capacity(nn.dhidden());

      for _ in (0..nn.dhidden()) { _v.push(0f64); }

      trainer.dinput.push(_v);
    }

    for _ in (0..nn.dhidden() + 1) {
      let mut _v = Vec::with_capacity(nn.doutput());

      for _ in (0..nn.doutput()) { _v.push(0f64); }

      trainer.doutput.push(_v);
      trainer.ehidden.push(0f64);
    }

    for _ in (0..nn.doutput() + 1) { trainer.eoutput.push(0f64); }

    trainer
  }
}