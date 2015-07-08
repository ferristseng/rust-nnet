//! Neural Nets

#![warn(missing_docs)]

#[macro_use] 
extern crate log;
extern crate num;
extern crate rand;
extern crate num_cpus;
extern crate threadpool;
extern crate rustc_serialize;

/// Implemented parameters for neural nets or trainers.
///
pub mod params;

/// Trait and enum definitions.
///
pub mod prelude;

/// Trainers. See module level documentation for detailed usage. 
///
pub mod trainer;