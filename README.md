# rust-nnet

_This library still isn't published to Cargo, nor would I consider it stable!_

Implementation of feed forward, single hidden layer, neural networks in Rust. 
Takes a macro-based approach to generate a neural network with fixed-size
arrays at compile time, following the assumption that the parameters
of a neural network are not likely to change.

  * `rust-nnet` - contains the trait definitions, and implementations of 
    trainers. 
  * `rust-nnet-macros` - contains the macros to generate the neural network 
    structure definition.
  * `rust-nnet-opencl` - contains trainer that utilizes opencl.

## examples

For a full example, see `examples/xor.rs`.

Add `rust-nnet` and `rust-nnet-macros` to your `Cargo.toml` file to use this 
library in your project.

```
[dependencies]
rust-nnet = "*"
rust-nnet-macros = "*"
```

Then import them into your project, and you're ready to go!

```rust
extern crate nnet;
#[macro_use(ffnn)] extern crate nnet_macros;
```

To create a feed forward neural network, you can call the `ffnn!` macro.
This macro will create a `new` function, and implement the `NeuralNet` 
trait defined in `nnet::prelude`.

The first parameter after the type identifier is the number of input 
nodes, the second the number of hidden nodes, and the last is the 
number of output nodes. 

```rust
ffnn!(XORNeuralNet, 2, 3, 1);
```

The macro will generate a type definition that looks like this:

```rust
pub struct XORNeuralNet<P> {
  input   : [f64; 3],
  hidden  : [f64; 4],
  output  : [f64; 1],
  winput  : [[f64; 3]; 3],
  woutput : [[f64; 1]; 4],
  ptype   : ::std::marker::PhantomData<P>
}
```

*The extra padding on the input and hidden layers are for the bias nodes.*

`P` is the parameter that is used to configure the neural network. The easiest 
way to start is just to configure it with some default parameters 
(sigmoid activation function, +1 bias nodes, and a default weight function).

```
use nnet::params::LogisticNeuralNet;
let mut nn: XORNeuralNet<LogisticNeuralNet> = LogisticNeuralNet::new();
```

It's pretty easy to define your own parameters, you just need to implement 
`ActivationFunction`, `WeightFunction`, and `BiasWeightFunction`, and 
`NeuralNetParameters` in `nnet::prelude`.

`nnet` comes with some trainers to adjust the weights of a NeuralNetwork given 
a training set. The training set can be defined as a slice of `(&[f64], &[f64])`
, or can be a slice of anything implementing `TrainingSetMember` 
in `nnet::prelude`.

If you wanted to train the neural network using backpropagation 
in a certain number of epochs, you could write:

```rust
use nnet::trainer::backpropagation::SeqEpochTrainer;
use nnet::prelude::{NeuralNetTrainer, NeuralNet, MomentumConstant, LearningRate};

struct MyTrainerParams;

impl MomentumConstant for MyTrainerParams {
  fn momentum() -> f64 { 0.8f64 }
}

impl LearningRate for MyTrainerParams {
  fn lrate() -> f64 { 0.3f64 }
}

...

let trainer: SeqEpochTrainer<_, _, MyTrainerParams, _> = 
  SeqEpochTrainer::with_epochs(&mut nn, &xor, 0.01, 5000);

for epoch in trainer {
  println!("Epoch: {:?}", epoch);
}
```

## license 

The MIT License (MIT)

Copyright (c) 2014-2015 Ferris Tseng

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.