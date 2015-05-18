use names::GENERIC_TYPE_PARAM;

use syntax::codemap::Span;
use syntax::ext::base::ExtCtxt;
use syntax::ext::build::AstBuilder;
use syntax::ast::{Ident, Path};


/// Creates a path to ::std::marker::PhantomData<P>.
#[inline] pub fn phantom_data_path(
  cx: &ExtCtxt, 
  sp: Span,
  gen: bool
) -> Path {
  cx.path_all(
    sp, 
    true, 
    vec![
      cx.ident_of_std("std"),
      cx.ident_of("marker"),
      cx.ident_of("PhantomData")
    ], 
    Vec::new(),
    if gen { 
      vec![cx.ty_ident(sp, cx.ident_of(GENERIC_TYPE_PARAM))]
    } else {
      Vec::new()
    },
    Vec::new())
}


// Creates a path to `nnet::params::NNParameters`.
#[inline] pub fn nnparameters_path(
  cx: &ExtCtxt,
  sp: Span
) -> Path {
  cx.path_all(
    sp, 
    true, 
    vec![
      cx.ident_of("nnet"),
      cx.ident_of("prelude"),
      cx.ident_of("NNParameters")
    ], 
    Vec::new(), 
    Vec::new(), 
    Vec::new())
}


#[inline] pub fn ffnn_path(
  cx: &ExtCtxt,
  sp: Span,
  name: Ident
) -> Path {
  cx.path_all(
    sp,
    false,
    vec![
      name
    ],
    Vec::new(),
    vec![cx.ty_ident(sp, cx.ident_of(GENERIC_TYPE_PARAM))],
    Vec::new())
}