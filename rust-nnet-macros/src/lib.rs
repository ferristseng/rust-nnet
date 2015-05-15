#![feature(rustc_private, plugin_registrar, slice_patterns, box_syntax)]

extern crate syntax;
extern crate rustc;

mod names;

use std::str::FromStr;

use names::*;
use syntax::ptr::P;
use syntax::codemap::{Span, Spanned};
use syntax::ast::{TokenTree, TtToken, StructDef, Item, StructField, Expr,
  StructField_, Ty, Ident, Generics, WhereClause, Path, NodeId, ImplItem};
use syntax::ext::base::{ExtCtxt, MacResult, DummyResult, MacEager};
use syntax::ext::build::AstBuilder;
use syntax::owned_slice::OwnedSlice;
use syntax::util::small_vector::SmallVector;
use rustc::plugin::Registry;


/// Macro expander for `create_ffnn!(Ident, usize, usize, usize)`.
fn expand_create_ffnn(
  cx: &mut ExtCtxt, 
  sp: Span, 
  args: &[TokenTree]
) -> Box<MacResult + 'static> {
  use syntax::parse::token::Lit::*;
  use syntax::parse::token::Token::*;

  let (name, num_in, num_hi, num_ou) = match args {
    [
      TtToken(_, Ident(name, _)), 
      TtToken(_, Comma), 
      TtToken(_, Literal(Integer(num_in), _)),
      TtToken(_, Comma),
      TtToken(_, Literal(Integer(num_hi), _)),
      TtToken(_, Comma),
      TtToken(_, Literal(Integer(num_ou), _))
    ] => {
      macro_rules! convert_to_usize (
        ($n:expr, $e:expr) => {
          match FromStr::from_str($n.as_str()) {
            Ok(i) if i > 0 => i,
            _ => {
              cx.span_err(sp, $e);
              return DummyResult::any(sp);
            }
          }
        }
      );

      let _num_in: usize = convert_to_usize!(
        num_in, 
        "number of input nodes must be a usize > 0");
      let _num_hi: usize = convert_to_usize!(
        num_hi, 
        "number of hidden nodes must be a usize > 0");
      let _num_ou: usize = convert_to_usize!(
        num_ou,
        "number of output nodes must be a usize > 0");

      (name, _num_in, _num_hi, _num_ou)
    }
    _ => {
      cx.span_err(sp, "invalid arguments...should be (Ident, usize, usize, usize)");
      return DummyResult::any(sp);
    }
  };

  MacEager::items(SmallVector::many(vec![
    create_ffnn_struct_def(cx, sp, name, num_in, num_hi, num_ou),
    impl_ffnn_new(cx, sp, name, num_in, num_ou),
    use_nnet_prelude(cx, sp)  
  ]))
}


/// use `nnet::prelude::*` for traits.
fn use_nnet_prelude(cx: &ExtCtxt, sp: Span) -> P<Item> {
  use syntax::ast::Visibility;

  cx.item_use_glob(
    sp, 
    Visibility::Inherited, 
    vec![
      cx.ident_of("nnet"),
      cx.ident_of("prelude")
    ])
}


/// Implements the new function for the neural network.
fn impl_ffnn_new(
  cx: &ExtCtxt, 
  sp: Span, 
  name: Ident,
  ins: usize, 
  outs: usize,
) -> P<Item> {
  use syntax::abi::Abi;
  use syntax::ast::Item_::*;
  use syntax::ast::ImplItem_::*;
  use syntax::ast::{Unsafety, ImplPolarity, MethodSig, ExplicitSelf_, Visibility};

  let nodeid = sp.expn_id.to_llvm_cookie() as u32;

  P(
    Item {
      ident: name,
      span: sp,
      attrs: Vec::new(),
      id: nodeid,
      node: ItemImpl(
        Unsafety::Normal,
        ImplPolarity::Positive, 
        Generics {
          lifetimes: Vec::new(),
          ty_params: OwnedSlice::from_vec(vec![
            cx.typaram(
              sp, 
              cx.ident_of(GENERIC_TYPE_PARAM),
              OwnedSlice::from_vec(vec![cx.typarambound(nnparameters_path(cx, sp))]), 
              None)]),
          where_clause: WhereClause { id: nodeid, predicates: Vec::new() },
        },
        None,
        cx.ty_path(ffnn_path(cx, sp, name)),
        vec![
          P(ImplItem {
            id: nodeid,
            span: sp,
            ident: cx.ident_of("new"),
            vis: Visibility::Public,
            attrs: Vec::new(),
            node: MethodImplItem(
              MethodSig {
                unsafety: Unsafety::Normal,
                abi: Abi::Rust,
                decl: cx.fn_decl(Vec::new(), cx.ty_path(ffnn_path(cx, sp, name))),
                generics: Generics {
                  lifetimes: Vec::new(),
                  ty_params: OwnedSlice::empty(),
                  where_clause: WhereClause { 
                    id: nodeid, 
                    predicates: Vec::new() 
                  }
                },
                explicit_self: Spanned { 
                  node: ExplicitSelf_::SelfStatic, 
                  span: sp 
                }
              },
              cx.block(
                sp, 
                Vec::new(), 
                Some(cx.expr_struct(
                  sp, 
                  ffnn_path(cx, sp, name), 
                  vec![
                    cx.field_imm(
                      sp, 
                      cx.ident_of(INPUT_FIELD_NAME), 
                      cx.expr_vec(sp, vec![
                        f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)),
                        f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)),
                        call_bias_function(cx, sp)
                      ])),
                    cx.field_imm(
                      sp,
                      cx.ident_of(PHANTOM_DATA_FIELD_NAME),
                      cx.expr_path(phantom_data_path(cx, sp, false))),
                    cx.field_imm(
                      sp,
                      cx.ident_of(OUTPUT_FIELD_NAME),
                      cx.expr_vec(
                        sp, 
                        vec![f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO))])),
                    cx.field_imm(
                      sp, 
                      cx.ident_of(HIDDEN_FIELD_NAME), 
                      cx.expr_vec(sp, vec![
                        f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)),
                        f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)),
                        f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)),
                        call_bias_function(cx, sp)
                      ])),
                    cx.field_imm(
                      sp,
                      cx.ident_of(WEIGHT_INPUT_HIDDEN_FIELD_NAME),
                      cx.expr_vec(sp, vec![
                        cx.expr_vec(sp,
                          vec![
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs)
                          ]),
                        cx.expr_vec(sp,
                          vec![
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs)
                          ]),
                        cx.expr_vec(sp,
                          vec![
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs),
                            call_weight_function(cx, sp, ins, outs)
                          ])
                      ]))
                  ]))))
          })
        ]
      ),
      vis: Visibility::Inherited
    })
}


fn f64_literal(cx: &ExtCtxt, sp: Span, ident: Ident) -> P<Expr> {
  use syntax::ast::{Lit_, FloatTy};
  use syntax::parse::token;

  cx.expr_lit(
    sp, 
    Lit_::LitFloat(token::get_ident(ident), FloatTy::TyF64))
}


fn call_bias_function(cx: &ExtCtxt, sp: Span) -> P<Expr> {
  cx.expr_call(
    sp,
    cx.expr_path(
      cx.path_all(
        sp,
        false,
        vec![
          cx.ident_of(GENERIC_TYPE_PARAM),
          cx.ident_of(BIAS_FUNCTION_TRAIT_NAME),
          cx.ident_of(BIAS_FUNCTION_NAME)
        ],
        Vec::new(),
        Vec::new(),
        Vec::new())),
    Vec::new())
}


fn call_weight_function(
  cx: &ExtCtxt, 
  sp: Span,
  ins: usize, 
  outs: usize
) -> P<Expr> {
  cx.expr_call(
    sp,
    cx.expr_path(
      cx.path_all(
        sp, false,
        vec![
          cx.ident_of(GENERIC_TYPE_PARAM),
          cx.ident_of(WEIGHT_FUNCTION_TRAIT_NAME),
          cx.ident_of(WEIGHT_FUNCTION_NAME)
        ],
        Vec::new(),
        Vec::new(),
        Vec::new())),
    vec![cx.expr_usize(sp, ins), cx.expr_usize(sp, outs)])
}


/// Creates a generic fixed array field for a struct.
fn create_fixed_array_struct_field(
  cx: &ExtCtxt, 
  id: NodeId,
  sp: Span, 
  ident: Ident,
  kind: P<Ty>,
  size: usize,
) -> StructField {
  use syntax::ast::{StructFieldKind, Visibility, Ty_};

  Spanned {
    node: StructField_ {
      kind: StructFieldKind::NamedField(ident, Visibility::Inherited),
      id: id,
      ty: cx.ty(
        sp, 
        Ty_::TyFixedLengthVec(
          kind, 
          cx.expr_usize(sp, size))),
      attrs: Vec::new()
    },
    span: sp
  }
}


/// Creates a path to ::std::marker::PhantomData<P>.
#[inline] fn phantom_data_path(
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
#[inline] fn nnparameters_path(
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


#[inline] fn ffnn_path(
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


/// Creates the structure definition for the feed forward
fn create_ffnn_struct_def(
  cx: &ExtCtxt, 
  sp: Span, 
  name: Ident,
  num_in: usize,
  num_hi: usize,
  num_ou: usize
) -> P<Item> {
  use syntax::ast::StructFieldKind::*;
  use syntax::ast::{Visibility, Ty_};

  let nodeid = sp.expn_id.to_llvm_cookie() as u32;
  let ffnn = StructDef { 
    fields: vec![
      Spanned {
        node: StructField_ {
          kind: NamedField(cx.ident_of(PHANTOM_DATA_FIELD_NAME), Visibility::Inherited),
          id: nodeid,
          ty: cx.ty_path(phantom_data_path(cx, sp, true)),
          attrs: Vec::new()
        },
        span: sp
      },
      create_fixed_array_struct_field(cx, nodeid, sp, 
        cx.ident_of(INPUT_FIELD_NAME), cx.ty_ident(sp, cx.ident_of(FLOAT_TYPE)),
        num_in + 1),
      create_fixed_array_struct_field(cx, nodeid, sp, 
        cx.ident_of(OUTPUT_FIELD_NAME), cx.ty_ident(sp, cx.ident_of(FLOAT_TYPE)),
        num_ou),
      create_fixed_array_struct_field(cx, nodeid, sp, 
        cx.ident_of(HIDDEN_FIELD_NAME), cx.ty_ident(sp, cx.ident_of(FLOAT_TYPE)),
        num_hi + 1),
      create_fixed_array_struct_field(cx, nodeid, sp,
        cx.ident_of(WEIGHT_INPUT_HIDDEN_FIELD_NAME),
        cx.ty(sp, Ty_::TyFixedLengthVec(
          cx.ty_ident(sp, cx.ident_of(FLOAT_TYPE)),
          cx.expr_usize(sp, num_hi))),
        num_in + 1),
      create_fixed_array_struct_field(cx, nodeid, sp, 
        cx.ident_of(WEIGHT_HIDDEN_OUTPUT_FIELD_NAME), 
        cx.ty(sp, Ty_::TyFixedLengthVec(
          cx.ty_ident(sp, cx.ident_of(FLOAT_TYPE)), 
          cx.expr_usize(sp, num_ou))),
        num_hi + 1)
    ], 
    ctor_id: None 
  };

  let generics = Generics { 
    lifetimes: Vec::new(),
    ty_params: OwnedSlice::from_vec(vec![
      cx.typaram(
        sp, 
        cx.ident_of(GENERIC_TYPE_PARAM), OwnedSlice::empty(), None)
      ]),
    where_clause: WhereClause { id: nodeid, predicates: Vec::new() }
  };

  cx.item_struct_poly(sp, name, ffnn, generics)
}


#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
  reg.register_macro("create_ffnn", expand_create_ffnn);
}