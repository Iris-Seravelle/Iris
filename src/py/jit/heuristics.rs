// src/py/jit/heuristics.rs
//! Placeholder heuristics module for expression simplification and optimization.
//!
//! Currently this is a no-op; future improvements may perform constant folding,
//! algebraic simplifications, loop unrolling hints, branch prediction hints, etc.

use crate::py::jit::parser::Expr;

/// Try to express `expr` as `a*var + b` where `var` is the loop variable.
/// Returns coefficients `(a,b)` if successful.
fn extract_linear(expr: &Expr, var: &str) -> Option<(f64, f64)> {
    match expr {
        Expr::Var(v) if v == var => Some((1.0, 0.0)),
        Expr::Const(c) => Some((0.0, *c)),
        Expr::BinOp(l, op, r) => match op.as_str() {
            "+" => {
                if let (Some((a1,b1)), Some((a2,b2))) = (extract_linear(l, var), extract_linear(r, var)) {
                    Some((a1 + a2, b1 + b2))
                } else {
                    None
                }
            }
            "-" => {
                if let (Some((a1,b1)), Some((a2,b2))) = (extract_linear(l, var), extract_linear(r, var)) {
                    Some((a1 - a2, b1 - b2))
                } else {
                    None
                }
            }
            "*" => {
                // one side must be constant
                if let Expr::Const(c) = **l {
                    if let Some((a,b)) = extract_linear(r, var) {
                        return Some((a * c, b * c));
                    }
                }
                if let Expr::Const(c) = **r {
                    if let Some((a,b)) = extract_linear(l, var) {
                        return Some((a * c, b * c));
                    }
                }
                None
            }
            _ => None,
        },
        Expr::UnaryOp('-', e) => {
            extract_linear(e, var).map(|(a,b)| (-a, -b))
        }
        _ => None,
    }
}

/// Try to express `expr` as `a*var*var + b*var + c`.
/// Returns coefficients `(a,b,c)` if successful.
fn extract_quadratic(expr: &Expr, var: &str) -> Option<(f64, f64, f64)> {
    match expr {
        Expr::Var(v) if v == var => Some((0.0, 1.0, 0.0)),
        Expr::Const(c) => Some((0.0, 0.0, *c)),
        Expr::BinOp(l, op, r) => match op.as_str() {
            "+" => {
                if let (Some((a1,b1,c1)), Some((a2,b2,c2))) = (extract_quadratic(l, var), extract_quadratic(r, var)) {
                    Some((a1 + a2, b1 + b2, c1 + c2))
                } else {
                    None
                }
            }
            "-" => {
                if let (Some((a1,b1,c1)), Some((a2,b2,c2))) = (extract_quadratic(l, var), extract_quadratic(r, var)) {
                    Some((a1 - a2, b1 - b2, c1 - c2))
                } else {
                    None
                }
            }
            "*" => {
                // direct square of the loop variable
                if **l == Expr::Var(var.to_string()) && **r == Expr::Var(var.to_string()) {
                    return Some((1.0, 0.0, 0.0));
                }
                // one side constant, the other quadratic
                if let Expr::Const(c) = **l {
                    if let Some((a,b,c0)) = extract_quadratic(r, var) {
                        return Some((a * c, b * c, c0 * c));
                    }
                }
                if let Expr::Const(c) = **r {
                    if let Some((a,b,c0)) = extract_quadratic(l, var) {
                        return Some((a * c, b * c, c0 * c));
                    }
                }
                None
            }
            _ => None,
        },
        Expr::UnaryOp('-', e) => {
            extract_quadratic(e, var).map(|(a,b,c)| (-a, -b, -c))
        }
        _ => None,
    }
}

// build closed-form expression for sum_{i=0}^{n-1} (a*i^2 + b*i + c)
fn build_quadratic_sum_formula(a: f64, b: f64, c: f64, n: Expr) -> Expr {
    let f = |v: f64| Expr::Const(v);
    let mul = |l: Expr, r: Expr| Expr::BinOp(Box::new(l), "*".to_string(), Box::new(r));
    let add = |l: Expr, r: Expr| Expr::BinOp(Box::new(l), "+".to_string(), Box::new(r));

    let nm1 = add(n.clone(), Expr::UnaryOp('-', Box::new(f(1.0))));
    let sum_i = mul(n.clone(), nm1.clone());
    let sum_i = Expr::BinOp(Box::new(sum_i), "/".to_string(), Box::new(f(2.0)));

    let two_n = mul(f(2.0), n.clone());
    let two_n_m1 = add(two_n, Expr::UnaryOp('-', Box::new(f(1.0))));
    let sum_i2 = mul(nm1.clone(), mul(n.clone(), two_n_m1));
    let sum_i2 = Expr::BinOp(Box::new(sum_i2), "/".to_string(), Box::new(f(6.0)));

    let part_a = mul(f(a), sum_i2);
    let part_b = mul(f(b), sum_i);
    let part_c = mul(f(c), n);
    add(add(part_a, part_b), part_c)
}

/// Apply lightweight transformations to an expression before code generation.
///
/// Right now it simply returns the input unchanged, but having a dedicated
/// module makes it easy to incrementally introduce more intelligence without
/// cluttering parser/codegen logic.
pub fn optimize(expr: Expr) -> Expr {
    // constant folding, algebraic simplifications and loop rewrites
    match expr {
        Expr::BinOp(lhs, op, rhs) => {
            let lhs = optimize(*lhs);
            let rhs = optimize(*rhs);

            // constant folding first
            if let (Expr::Const(a), Expr::Const(b)) = (&lhs, &rhs) {
                let v = match op.as_str() {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    "**" => a.powf(*b),
                    _ => return Expr::BinOp(Box::new(lhs), op, Box::new(rhs)),
                };
                return Expr::Const(v);
            }

            // algebraic simplifications
            // simple algebraic simplifications for zeros/ones
            if op == "+" && matches!(lhs, Expr::Const(0.0)) {
                return rhs.clone();
            }
            if op == "+" && matches!(rhs, Expr::Const(0.0)) {
                return lhs.clone();
            }
            if op == "-" && matches!(rhs, Expr::Const(0.0)) {
                return lhs.clone();
            }
            if op == "*" && (matches!(lhs, Expr::Const(0.0)) || matches!(rhs, Expr::Const(0.0))) {
                return Expr::Const(0.0);
            }
            if op == "*" && matches!(lhs, Expr::Const(1.0)) {
                return rhs.clone();
            }
            if op == "*" && matches!(rhs, Expr::Const(1.0)) {
                return lhs.clone();
            }
            if op == "/" && matches!(rhs, Expr::Const(1.0)) {
                return lhs.clone();
            }

            // more advanced equalities
            if lhs == rhs {
                match op.as_str() {
                    "-" => return Expr::Const(0.0),
                    "/" => return Expr::Const(1.0),
                    _ => {}
                }
            }
            if op == "+" {
                if let Expr::UnaryOp('-', inner) = &rhs {
                    if **inner == lhs {
                        return Expr::Const(0.0);
                    }
                }
            }

            Expr::BinOp(Box::new(lhs), op, Box::new(rhs))
        }
        Expr::UnaryOp(c, expr) => {
            let expr = optimize(*expr);
            // eliminate double-negatives
            if let Expr::UnaryOp('-', inner) = &expr {
                if c == '-' {
                    return *inner.clone();
                }
            }
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
            if let Expr::Const(c) = &cond {
                return if *c != 0.0 { thenb } else { elseb };
            }
            Expr::Ternary(Box::new(cond), Box::new(thenb), Box::new(elseb))
        }
        Expr::Call(name, args) => {
            // optimize arguments first
            let args = args.into_iter().map(optimize).collect::<Vec<_>>();
            // constant propagation for known math calls
            if args.iter().all(|e| matches!(e, Expr::Const(_))) {
                if let Some(val) = evaluate_call(&name, &args) {
                    return Expr::Const(val);
                }
            }
            Expr::Call(name, args)
        }
        Expr::SumFor { iter_var, start, end, body } => {
            let start = optimize(*start);
            let end = optimize(*end);
            let body = optimize(*body);

            // quadratic rewrite when range starts at zero and body is quadratic
            if matches!(start, Expr::Const(0.0)) {
                if let Some((a_coeff, b_coeff, c_coeff)) = extract_quadratic(&body, &iter_var) {
                    eprintln!("quadratic detected coeffs: {:?}, body={:?}", (a_coeff, b_coeff, c_coeff), body);
                    // rewrite using formula with n = end
                    let formula = build_quadratic_sum_formula(a_coeff, b_coeff, c_coeff, end.clone());
                    return optimize(formula);
                }
            }

            // handle simple coefficient-of-variable pattern: i * k or k * i
            if let Expr::BinOp(lc, op, rc) = &body {
                if op == "*" {
                    if **lc == Expr::Var(iter_var.clone()) {
                        let coeff = *rc.clone();
                        let sum_i = Expr::SumFor {
                            iter_var: iter_var.clone(),
                            start: Box::new(start.clone()),
                            end: Box::new(end.clone()),
                            body: Box::new(Expr::Var(iter_var.clone())),
                        };
                        return optimize(Expr::BinOp(Box::new(coeff), "*".to_string(), Box::new(sum_i)));
                    }
                    if **rc == Expr::Var(iter_var.clone()) {
                        let coeff = *lc.clone();
                        let sum_i = Expr::SumFor {
                            iter_var: iter_var.clone(),
                            start: Box::new(start.clone()),
                            end: Box::new(end.clone()),
                            body: Box::new(Expr::Var(iter_var.clone())),
                        };
                        return optimize(Expr::BinOp(Box::new(coeff), "*".to_string(), Box::new(sum_i)));
                    }
                }
            }

            // try to decompose nontrivial linear body a*i + b
            if !matches!(body, Expr::Var(_) | Expr::Const(_)) {
                if let Some((a_coeff, b_const)) = extract_linear(&body, &iter_var) {
                    let var = iter_var.clone();
                    let sum_i = Expr::SumFor {
                        iter_var: var.clone(),
                        start: Box::new(start.clone()),
                        end: Box::new(end.clone()),
                        body: Box::new(Expr::Var(var.clone())),
                    };
                    let sum_one = Expr::SumFor {
                        iter_var: var.clone(),
                        start: Box::new(start.clone()),
                        end: Box::new(end.clone()),
                        body: Box::new(Expr::Const(1.0)),
                    };
                    let rewritten = Expr::BinOp(
                        Box::new(Expr::BinOp(
                            Box::new(Expr::Const(a_coeff)),
                            "*".to_string(),
                            Box::new(sum_i),
                        )),
                        "+".to_string(),
                        Box::new(Expr::BinOp(
                            Box::new(Expr::Const(b_const)),
                            "*".to_string(),
                            Box::new(sum_one),
                        )),
                    );
                    return optimize(rewritten);
                }
            }

            // if bounds are constant we can sometimes reduce the loop
            if let (Expr::Const(a), Expr::Const(b)) = (&start, &end) {
                let len = b - a; // number of iterations (exclusive end)
                if len.is_finite() {
                    // case 1: body is just the loop variable -> arithmetic series
                    if body == Expr::Var(iter_var.clone()) {
                        // sum over i=a..b-1 = n*(a + (b-1))/2
                        let sum = len * (a + (b - 1.0)) / 2.0;
                        return Expr::Const(sum);
                    }
                    // case 2: body is constant -> constant*len
                    if let Expr::Const(c) = body {
                        return Expr::Const(c * len);
                    }
                    // general constant evaluation using interpreter
                    if let (Some(a0), Some(b0)) = (eval_const(&start), eval_const(&end)) {
                        let mut acc = 0.0;
                        for i in (a0 as i64)..(b0 as i64) {
                            let mut env = std::collections::HashMap::new();
                            env.insert(iter_var.clone(), i as f64);
                            if let Some(v) = eval_expr(&body, &env) {
                                acc += v;
                            } else {
                                acc = std::f64::NAN;
                                break;
                            }
                        }
                        if acc.is_finite() {
                            return Expr::Const(acc);
                        }
                    }
                }
            }

            Expr::SumFor {
                iter_var,
                start: Box::new(start),
                end: Box::new(end),
                body: Box::new(body),
            }
        },
        other => other,
    }
}


fn evaluate_call(name: &str, args: &[Expr]) -> Option<f64> {
    // very simple interpreter for a handful of math functions
    let vals: Vec<f64> = args
        .iter()
        .map(|e| if let Expr::Const(v) = e { *v } else { 0.0 })
        .collect();
    match (name, vals.as_slice()) {
        ("sin", [x]) => Some(x.sin()),
        ("cos", [x]) => Some(x.cos()),
        ("tan", [x]) => Some(x.tan()),
        ("sinh", [x]) => Some(x.sinh()),
        ("cosh", [x]) => Some(x.cosh()),
        ("tanh", [x]) => Some(x.tanh()),
        ("exp", [x]) => Some(x.exp()),
        ("log", [x]) => Some(x.ln()),
        ("sqrt", [x]) => Some(x.sqrt()),
        ("abs", [x]) => Some(x.abs()),
        ("pow", [x, y]) => Some(x.powf(*y)),
        _ => None,
    }
}

/// Evaluate an expression with concrete variable bindings; returns `None` if
/// it contains an unbound variable or unsupported construct.
fn eval_expr(expr: &Expr, env: &std::collections::HashMap<String, f64>) -> Option<f64> {
    match expr {
        Expr::Const(v) => Some(*v),
        Expr::Var(v) => env.get(v).cloned(),
        Expr::BinOp(l, op, r) => {
            let a = eval_expr(l, env)?;
            let b = eval_expr(r, env)?;
            match op.as_str() {
                "+" => Some(a + b),
                "-" => Some(a - b),
                "*" => Some(a * b),
                "/" => Some(a / b),
                "**" => Some(a.powf(b)),
                _ => None,
            }
        }
        Expr::UnaryOp(c, e) => {
            let v = eval_expr(e, env)?;
            if *c == '-' { Some(-v) } else { Some(v) }
        }
        Expr::Call(name, args) => {
            let vals: Vec<Expr> = args.iter().filter_map(|e| {
                eval_expr(e, env).map(Expr::Const)
            }).collect();
            evaluate_call(name, &vals)
        }
        _ => None,
    }
}

fn eval_const(expr: &Expr) -> Option<f64> {
    if let Expr::Const(v) = expr {
        Some(*v)
    } else {
        None
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

    #[test]
    fn algebraic_simplifications() {
        assert_eq!(optimize(parse("x + 0")), parse("x"));
        assert_eq!(optimize(parse("0 + x")), parse("x"));
        assert_eq!(optimize(parse("x - 0")), parse("x"));
        assert_eq!(optimize(parse("x * 1")), parse("x"));
        assert_eq!(optimize(parse("1 * y")), parse("y"));
        assert_eq!(optimize(parse("y * 0")), Expr::Const(0.0));
        assert_eq!(optimize(parse("0 * y")), Expr::Const(0.0));
        assert_eq!(optimize(parse("x / 1")), parse("x"));
    }

    #[test]
    fn ternary_constants() {
        assert_eq!(optimize(parse("x if 1 else y")), parse("x"));
        assert_eq!(optimize(parse("x if 0 else y")), parse("y"));
    }

    #[test]
    fn sum_for_reduction() {
        assert_eq!(optimize(parse("sum(i for i in range(5))")), Expr::Const(10.0));
        assert_eq!(optimize(parse("sum(2 for i in range(3))")), Expr::Const(6.0));
        assert_eq!(optimize(parse("sum(i for i in range(2,5))")), Expr::Const(9.0));
    }

    #[test]
    fn algebraic_advanced() {
        assert_eq!(optimize(parse("x - x")), Expr::Const(0.0));
        assert_eq!(optimize(parse("x + (-x)")), Expr::Const(0.0));
        assert_eq!(optimize(parse("x / x")), Expr::Const(1.0));
        assert_eq!(optimize(parse("-(-x)")), parse("x"));
    }

    #[test]
    fn const_math_calls() {
        assert_eq!(optimize(parse("sin(0)")), Expr::Const(0.0));
        assert_eq!(optimize(parse("cos(0)")), Expr::Const(1.0));
        assert_eq!(optimize(parse("exp(1)")), Expr::Const(std::f64::consts::E));
        assert_eq!(optimize(parse("pow(2,3)")), Expr::Const(8.0));
    }

    #[test]
    fn linear_sum_constant_bounds() {
        // sum(2*i + 3 for i in range(0,4)) -> compute constants
        assert_eq!(
            optimize(parse("sum(2*i + 3 for i in range(0,4))")),
            Expr::Const(2.0 * 6.0 + 3.0 * 4.0)
        );
    }

    #[test]
    fn linear_sum_variable_bounds() {
        let expr = parse("sum(2*i + 3 for i in range(n))");
        let out = optimize(expr.clone());
        eprintln!("linear var bounds: {:?} -> {:?}", expr, out);
        // should at least change
        assert_ne!(out, expr);
    }

    #[test]
    fn quadratic_rewrites() {
        let expr = parse("sum(i*i for i in range(n))");
        let out = optimize(expr.clone());
        eprintln!("quadratic var bounds: {:?} -> {:?}", expr, out);
        assert_ne!(out, expr);
        // evaluate for a few n to ensure equivalence
        let mut env = std::collections::HashMap::new();
        for n in 0..5 {
            env.insert("n".to_string(), n as f64);
            let orig = {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += (i as f64) * (i as f64);
                }
                sum
            };
            let eval = eval_expr(&out, &env).unwrap();
            assert!((orig - eval).abs() < 1e-9);
        }
    }

    #[test]
    fn constant_bound_general() {
        let expr = parse("sum((i-1)*(i+1) for i in range(0,5))");
        let out = optimize(expr.clone());
        // computed manually: (-1)+0+3+8+15 = 25
        assert_eq!(out, Expr::Const(25.0));
    }
}
