#[macro_export]
macro_rules! ffnn {
  ([$($mt:meta),*]; $ty:ident, $inputs:expr, $hidden:expr, $outputs:expr) => (
    $(#[$mt])*
    #[derive(Clone)]
    pub struct $ty<P> {
      input   : [f64; $inputs + 1],
      hidden  : [f64; $hidden + 1],
      output  : [f64; $outputs],
      winput  : [[f64; $hidden]; $inputs + 1],
      woutput : [[f64; $outputs]; $hidden + 1],
      ptype   : ::std::marker::PhantomData<P>
    }

    impl<P> $ty<P> where P : nnet::prelude::NeuralNetParameters {
      pub fn new() -> $ty<P> {
        use nnet::prelude::{WeightFunction, BiasWeightFunction};

        let mut _nn = $ty {
          input   : [0f64; $inputs + 1],
          hidden  : [0f64; $hidden + 1],
          output  : [0f64; $outputs],
          winput  : [[0f64; $hidden]; $inputs + 1],
          woutput : [[0f64; $outputs]; $hidden + 1],
          ptype   : ::std::marker::PhantomData
        };

        for ws in _nn.winput.iter_mut() {
          for w in ws.iter_mut() {
            *w = P::WeightFunction::initw($inputs, $outputs);
          }
        }

        for ws in _nn.woutput.iter_mut() {
          for w in ws.iter_mut() {
            *w = P::WeightFunction::initw($inputs, $outputs);
          }
        }

        _nn.input[$inputs] = P::BiasWeightFunction::biasw();
        _nn.hidden[$hidden] = P::BiasWeightFunction::biasw();

        _nn
      }
    }

    impl<P> nnet::prelude::NeuralNet<P> for $ty<P> 
      where P : nnet::prelude::NeuralNetParameters 
    {
      #[inline(always)] fn dim_input() -> usize { $inputs }

      #[inline(always)] fn dim_output() -> usize { $outputs }

      #[inline(always)] fn dim_hidden() -> usize { $hidden }

      #[inline] 
      fn node(&self, node: nnet::prelude::Node) -> f64 { 
        match node {
          nnet::prelude::Node::Input(i) => self.input[i],
          nnet::prelude::Node::Hidden(i) => self.hidden[i],
          nnet::prelude::Node::Output(i) => self.output[i],
          nnet::prelude::Node::WeightInputHidden(i, j) => self.winput[i][j],
          nnet::prelude::Node::WeightHiddenOutput(i, j) => self.woutput[i][j]
        }
      }

      #[inline] 
      fn node_mut(&mut self, node: nnet::prelude::Node) -> &mut f64 { 
        match node {
          nnet::prelude::Node::Input(i) => &mut self.input[i],
          nnet::prelude::Node::Hidden(i) => &mut self.hidden[i],
          nnet::prelude::Node::Output(i) => &mut self.output[i],
          nnet::prelude::Node::WeightInputHidden(i, j) => &mut self.winput[i][j],
          nnet::prelude::Node::WeightHiddenOutput(i, j) => &mut self.woutput[i][j]
        }
      }

      #[inline] 
      fn layer(&self, layer: nnet::prelude::Layer) -> &[f64] {
        match layer {
          nnet::prelude::Layer::Input => self.input.as_ref(),
          nnet::prelude::Layer::Hidden => self.hidden.as_ref(),
          nnet::prelude::Layer::Output => self.output.as_ref()
        }
      }

      fn predict(&mut self, inp: &[f64]) {
        use nnet::prelude::ActivationFunction;

        assert!(inp.len() == Self::dim_input());

        for i in (0..Self::dim_input()) {
          self.input[i] = inp[i];
        }

        for i in (0..Self::dim_hidden()) {
          self.hidden[i] = 0f64;

          for j in (0..Self::dim_input() + 1) {
            self.hidden[i] += self.input[j] * 
              self.node(nnet::prelude::Node::WeightInputHidden(j, i));
          }

          self.hidden[i] = 
            P::ActivationFunction::activation(
              self.node(nnet::prelude::Node::Hidden(i)));
        }

        for i in (0..Self::dim_output()) {
          self.output[i] = 0f64;

          for j in (0..Self::dim_hidden() + 1) {
            self.output[i] += self.hidden[j] * 
            self.node(nnet::prelude::Node::WeightHiddenOutput(j, i));
          }

          self.output[i] = 
            P::ActivationFunction::activation(self.output[i]);
        }
      }
    }
  );
  ($ty:ident, $inputs:expr, $hidden:expr, $outputs:expr) => (
    ffnn!([]; $ty, $inputs, $hidden, $outputs);
  )
}