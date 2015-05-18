use util::*;
use names::*;
use syntax::ptr::P;
use syntax::codemap::{Span, Spanned};
use syntax::abi::Abi;
use syntax::ext::base::ExtCtxt;
use syntax::ext::build::AstBuilder;
use syntax::ast::Item_::*;
use syntax::ast::ImplItem_::*;
use syntax::ast::{Ident, Expr, Unsafety, ImplPolarity, MethodSig, 
  ExplicitSelf_, Visibility, Generics, WhereClause, Item, ImplItem};
use syntax::owned_slice::OwnedSlice;


/// Implements the new function for the neural network.
///
/// Code generated: 
///
/// ```
/// fn new<P>() -> NeuralNet<P> where P : NNParameters {
///   ...
/// }
/// ```
pub fn impl_ffnn_new(
  cx: &ExtCtxt, 
  sp: Span, 
  name: Ident,
  ins: usize, 
  hids: usize,
  outs: usize,
) -> P<Item> {
  let nodeid = sp.expn_id.to_llvm_cookie() as u32;
  let mut input_vector_init = build_expr_vec(
    ins, 
    || f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)));
  let mut hidden_vector_init = build_expr_vec(
    hids,
    || f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)));
  let output_vector_init = build_expr_vec(
    outs,
    || f64_literal(cx, sp, cx.ident_of(FLOAT_ZERO)));
  let weights_input_hidden_vector_init = build_expr_vec(
    ins + 1,
    || cx.expr_vec(
      sp,
      build_expr_vec(hids, || call_weight_function(cx, sp, ins, outs))));
  let weights_hidden_output_vector_init = build_expr_vec(
    hids + 1,
    || cx.expr_vec(
      sp,
      build_expr_vec(outs, || call_weight_function(cx, sp, ins, outs))));

  input_vector_init.push(call_bias_function(cx, sp));
  hidden_vector_init.push(call_bias_function(cx, sp));

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
                      cx.expr_vec(sp, input_vector_init)),
                    cx.field_imm(
                      sp,
                      cx.ident_of(PHANTOM_DATA_FIELD_NAME),
                      cx.expr_path(phantom_data_path(cx, sp, false))),
                    cx.field_imm(
                      sp,
                      cx.ident_of(OUTPUT_FIELD_NAME),
                      cx.expr_vec(sp, output_vector_init)),
                    cx.field_imm(
                      sp,
                      cx.ident_of(HIDDEN_FIELD_NAME),
                      cx.expr_vec(sp, hidden_vector_init)),
                    cx.field_imm(
                      sp,
                      cx.ident_of(WEIGHT_INPUT_HIDDEN_FIELD_NAME),
                      cx.expr_vec(sp, weights_input_hidden_vector_init)),
                    cx.field_imm(
                      sp, 
                      cx.ident_of(WEIGHT_HIDDEN_OUTPUT_FIELD_NAME),
                      cx.expr_vec(sp, weights_hidden_output_vector_init))
                  ]))))
          })
        ]
      ),
      vis: Visibility::Inherited
    })
}


fn build_expr_vec<F>(
  size: usize,
  builder: F
) -> Vec<P<Expr>> where F : Fn() -> P<Expr> {
  let mut vec = Vec::new();

  for _ in 0..size { vec.push(builder()); }

  vec
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