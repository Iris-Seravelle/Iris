// src/py/jit/heuristics.rs
//! Placeholder heuristics module for expression simplification and optimization.
//!
//! Currently this is a no-op; future improvements may perform constant folding,
//! algebraic simplifications, loop unrolling hints, branch prediction hints, etc.

use crate::py::jit::parser::Expr;

/// Apply lightweight transformations to an expression before code generation.
///
/// Right now it simply returns the input unchanged, but having a dedicated
/// module makes it easy to incrementally introduce more intelligence without
/// cluttering parser/codegen logic.
pub fn optimize(expr: Expr) -> Expr {
    // constant folding and minor simplifications
    match expr {
        Expr::BinOp(lhs, op, rhs) => {
            let lhs = optimize(*lhs);
            let rhs = optimize(*rhs);
            if let (Expr::Const(a), Expr::Const(b)) = (&lhs, &rhs) {
                let v = match op.as_str() {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    "**" => a.powf(*b),
                    _ => return Expr::BinOp(Box::new(lhs), op, Box::new(rhs)),
                };
                Expr::Const(v)
            } else {
                Expr::BinOp(Box::new(lhs), op, Box::new(rhs))
            }
        }
        Expr::UnaryOp(c, expr) => {
            let expr = optimize(*expr);
            if let Expr::Const(a) = expr {
                if c == '-' {
                    Expr::Const(-a)
                } else {
                    Expr::UnaryOp(c, Box::new(Expr::Const(a)))
                }
            } else {
                Expr::UnaryOp(c, Box::new(expr))
            }
        }
        Expr::Ternary(cond, thenb, elseb) => {
            let cond = optimize(*cond);
            let thenb = optimize(*thenb);
            let elseb = optimize(*elseb);
            Expr::Ternary(Box::new(cond), Box::new(thenb), Box::new(elseb))
        }
        Expr::Call(name, args) => {
            Expr::Call(name, args.into_iter().map(optimize).collect())
        }
        Expr::SumFor { iter_var, start, end, body } => Expr::SumFor {
            iter_var,
            start: Box::new(optimize(*start)),
            end: Box::new(optimize(*end)),
            body: Box::new(optimize(*body)),
        },
        other => other,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::py::jit::parser::{tokenize, Parser};

    fn parse(expr: &str) -> Expr {
        Parser::new(tokenize(expr)).parse_expr().unwrap()
    }

    #[test]
    fn constant_folding_binops() {
        assert_eq!(optimize(parse("2 + 2")), Expr::Const(4.0));
        assert_eq!(optimize(parse("2 * 3")), Expr::Const(6.0));
        assert_eq!(optimize(parse("2 ** 3")), Expr::Const(8.0));
    }

    #[test]
    fn constant_folding_unary() {
        assert_eq!(optimize(parse("-5")), Expr::Const(-5.0));
    }

    #[test]
    fn no_fold_with_variable() {
        assert_eq!(optimize(parse("2 + x + 3")), parse("2 + x + 3"));
    }
}
