// src/py/jit/codegen/lowering.rs
//! Helpers for converting a parsed `Expr` into a lowered kernel representation.

use crate::py::jit::parser::Expr;

use crate::py::jit::codegen::{
    LoweredBinaryKernel, LoweredExpr, LoweredKernel, LoweredUnaryKernel,
};

pub(crate) fn arg_index_of_var(arg_names: &[String], var: &str) -> Option<usize> {
    arg_names.iter().position(|n| n == var)
}

pub(crate) fn normalize_intrinsic_name(raw: &str) -> &str {
    raw.rsplit('.').next().unwrap_or(raw)
}

pub(crate) fn detect_lowered_kernel(expr: &Expr, arg_names: &[String]) -> Option<LoweredKernel> {
    match expr {
        Expr::Var(name) => {
            let input = arg_index_of_var(arg_names, name)?;
            Some(LoweredKernel::Unary {
                op: LoweredUnaryKernel::Identity,
                input,
            })
        }
        Expr::UnaryOp('-', sub) => {
            if let Expr::Var(name) = sub.as_ref() {
                if let Some(input) = arg_index_of_var(arg_names, name) {
                    return Some(LoweredKernel::Unary {
                        op: LoweredUnaryKernel::Neg,
                        input,
                    });
                }
            }
            detect_lowered_expr(expr, arg_names).map(LoweredKernel::Expr)
        }
        Expr::Call(name, args) => {
            if args.len() == 1 {
                if let Expr::Var(var_name) = &args[0] {
                    if let Some(input) = arg_index_of_var(arg_names, var_name) {
                        let op = match normalize_intrinsic_name(name).to_ascii_lowercase().as_str() {
                            "abs" | "fabs" => Some(LoweredUnaryKernel::Abs),
                            "sin" => Some(LoweredUnaryKernel::Sin),
                            "cos" => Some(LoweredUnaryKernel::Cos),
                            "tan" => Some(LoweredUnaryKernel::Tan),
                            "exp" => Some(LoweredUnaryKernel::Exp),
                            "log" | "ln" => Some(LoweredUnaryKernel::Log),
                            "sqrt" => Some(LoweredUnaryKernel::Sqrt),
                            _ => None,
                        };
                        if let Some(op) = op {
                            return Some(LoweredKernel::Unary { op, input });
                        }
                    }
                }
            }
            detect_lowered_expr(expr, arg_names).map(LoweredKernel::Expr)
        }
        Expr::BinOp(lhs, op, rhs) => {
            if let (Expr::Var(lhs_name), Expr::Var(rhs_name)) = (lhs.as_ref(), rhs.as_ref()) {
                if let (Some(lhs_idx), Some(rhs_idx)) = (
                    arg_index_of_var(arg_names, lhs_name),
                    arg_index_of_var(arg_names, rhs_name),
                ) {
                    let op = match op.as_str() {
                        "+" => Some(LoweredBinaryKernel::Add),
                        "-" => Some(LoweredBinaryKernel::Sub),
                        "*" => Some(LoweredBinaryKernel::Mul),
                        "/" => Some(LoweredBinaryKernel::Div),
                        _ => None,
                    };
                    if let Some(op) = op {
                        return Some(LoweredKernel::Binary {
                            op,
                            lhs: lhs_idx,
                            rhs: rhs_idx,
                        });
                    }
                }
            }
            detect_lowered_expr(expr, arg_names).map(LoweredKernel::Expr)
        }
        _ => detect_lowered_expr(expr, arg_names).map(LoweredKernel::Expr),
    }
}

pub(crate) fn detect_lowered_expr(expr: &Expr, arg_names: &[String]) -> Option<LoweredExpr> {
    match expr {
        Expr::Const(v) => Some(LoweredExpr::Const(*v)),
        Expr::Var(name) => {
            if let Some(idx) = arg_index_of_var(arg_names, name) {
                return Some(LoweredExpr::Input(idx));
            }
            None
        }
        Expr::BinOp(lhs, op, rhs) => {
            let l = detect_lowered_expr(lhs, arg_names)?;
            let r = detect_lowered_expr(rhs, arg_names)?;
            match op.as_str() {
                "+" => Some(LoweredExpr::Add(Box::new(l), Box::new(r))),
                "-" => Some(LoweredExpr::Sub(Box::new(l), Box::new(r))),
                "*" => Some(LoweredExpr::Mul(Box::new(l), Box::new(r))),
                "/" => Some(LoweredExpr::Div(Box::new(l), Box::new(r))),
                "%" => Some(LoweredExpr::Mod(Box::new(l), Box::new(r))),
                "==" => Some(LoweredExpr::Eq(Box::new(l), Box::new(r))),
                "!=" => Some(LoweredExpr::Ne(Box::new(l), Box::new(r))),
                "<" => Some(LoweredExpr::Lt(Box::new(l), Box::new(r))),
                ">" => Some(LoweredExpr::Gt(Box::new(l), Box::new(r))),
                "<=" => Some(LoweredExpr::Le(Box::new(l), Box::new(r))),
                ">=" => Some(LoweredExpr::Ge(Box::new(l), Box::new(r))),
                "and" => Some(LoweredExpr::And(Box::new(l), Box::new(r))),
                "or" => Some(LoweredExpr::Or(Box::new(l), Box::new(r))),
                _ => None,
            }
        }
        Expr::UnaryOp(op, sub) => {
            let v = detect_lowered_expr(sub, arg_names)?;
            match op {
                '-' => Some(LoweredExpr::Neg(Box::new(v))),
                '!' => Some(LoweredExpr::Not(Box::new(v))),
                _ => None,
            }
        }
        Expr::Call(name, args) => {
            if args.len() == 1 {
                let inner = detect_lowered_expr(&args[0], arg_names)?;
                match normalize_intrinsic_name(name).to_ascii_lowercase().as_str() {
                    "abs" | "fabs" => Some(LoweredExpr::Abs(Box::new(inner))),
                    "sin" => Some(LoweredExpr::Sin(Box::new(inner))),
                    "cos" => Some(LoweredExpr::Cos(Box::new(inner))),
                    "tan" => Some(LoweredExpr::Tan(Box::new(inner))),
                    "exp" => Some(LoweredExpr::Exp(Box::new(inner))),
                    "log" | "ln" => Some(LoweredExpr::Log(Box::new(inner))),
                    "sqrt" => Some(LoweredExpr::Sqrt(Box::new(inner))),
                    _ => None,
                }
            } else {
                None
            }
        }
        Expr::Ternary(cond, then_expr, else_expr) => Some(LoweredExpr::Ternary {
            cond: Box::new(detect_lowered_expr(cond, arg_names)?),
            then_expr: Box::new(detect_lowered_expr(then_expr, arg_names)?),
            else_expr: Box::new(detect_lowered_expr(else_expr, arg_names)?),
        }),
        _ => None,
    }
}
