// src/py/jit/codegen/eval.rs

use super::{
    fast_cos_approx, fast_sin_approx, read_buffer_f64, BufferView, LoweredBinaryKernel,
    LoweredExpr, LoweredUnaryKernel, SimdMathMode,
};

#[cfg(target_arch = "aarch64")]
use super::{fast_cos_approx_pair_neon, fast_sin_approx_pair_neon};

#[inline(always)]
pub(crate) fn lowered_unary_eval_pair(
    op: LoweredUnaryKernel,
    x0: f64,
    x1: f64,
    mode: SimdMathMode,
) -> Option<(f64, f64)> {
    if mode != SimdMathMode::FastApprox {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return match op {
            LoweredUnaryKernel::Sin => Some(fast_sin_approx_pair_neon(x0, x1)),
            LoweredUnaryKernel::Cos => Some(fast_cos_approx_pair_neon(x0, x1)),
            LoweredUnaryKernel::Tan => {
                let (s0, s1) = fast_sin_approx_pair_neon(x0, x1);
                let (c0, c1) = fast_cos_approx_pair_neon(x0, x1);
                Some((s0 / c0, s1 / c1))
            }
            _ => None,
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (op, x0, x1);
        None
    }
}

#[inline(always)]
pub(crate) fn lowered_unary_eval(op: LoweredUnaryKernel, x: f64, mode: SimdMathMode) -> f64 {
    match op {
        LoweredUnaryKernel::Identity => x,
        LoweredUnaryKernel::Neg => -x,
        LoweredUnaryKernel::Abs => x.abs(),
        LoweredUnaryKernel::Sin => {
            if mode == SimdMathMode::FastApprox {
                fast_sin_approx(x)
            } else {
                x.sin()
            }
        }
        LoweredUnaryKernel::Cos => {
            if mode == SimdMathMode::FastApprox {
                fast_cos_approx(x)
            } else {
                x.cos()
            }
        }
        LoweredUnaryKernel::Tan => {
            if mode == SimdMathMode::FastApprox {
                fast_sin_approx(x) / fast_cos_approx(x)
            } else {
                x.tan()
            }
        }
        LoweredUnaryKernel::Exp => x.exp(),
        LoweredUnaryKernel::Log => x.ln(),
        LoweredUnaryKernel::Sqrt => x.sqrt(),
    }
}

#[inline(always)]
pub(crate) fn lowered_binary_eval(op: LoweredBinaryKernel, lhs: f64, rhs: f64) -> f64 {
    match op {
        LoweredBinaryKernel::Add => lhs + rhs,
        LoweredBinaryKernel::Sub => lhs - rhs,
        LoweredBinaryKernel::Mul => lhs * rhs,
        LoweredBinaryKernel::Div => lhs / rhs,
    }
}

pub(crate) fn lowered_expr_eval(
    expr: &LoweredExpr,
    views: &[BufferView],
    idx: usize,
    mode: SimdMathMode,
) -> Option<f64> {
    match expr {
        LoweredExpr::Const(v) => Some(*v),
        LoweredExpr::Input(input_idx) => {
            let view = views.get(*input_idx)?;
            Some(unsafe { read_buffer_f64(view, idx) })
        }
        LoweredExpr::Add(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? + lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Sub(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? - lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Mul(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? * lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Div(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? / lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Mod(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? % lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Eq(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? == lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Ne(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Lt(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? < lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Gt(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? > lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Le(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? <= lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Ge(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? >= lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::And(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != 0.0
                && lowered_expr_eval(b, views, idx, mode)? != 0.0
            {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Or(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != 0.0
                || lowered_expr_eval(b, views, idx, mode)? != 0.0
            {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Neg(a) => Some(-lowered_expr_eval(a, views, idx, mode)?),
        LoweredExpr::Not(a) => Some(if lowered_expr_eval(a, views, idx, mode)? == 0.0 {
            1.0
        } else {
            0.0
        }),
        LoweredExpr::Abs(a) => Some(lowered_expr_eval(a, views, idx, mode)?.abs()),
        LoweredExpr::Sin(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Sin,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Cos(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Cos,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Tan(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Tan,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Exp(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Exp,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Log(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Log,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Sqrt(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Sqrt,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            if lowered_expr_eval(cond, views, idx, mode)? != 0.0 {
                lowered_expr_eval(then_expr, views, idx, mode)
            } else {
                lowered_expr_eval(else_expr, views, idx, mode)
            }
        }
    }
}

#[inline(always)]
pub(crate) fn lowered_unary_eval_pair_any(
    op: LoweredUnaryKernel,
    x0: f64,
    x1: f64,
    mode: SimdMathMode,
) -> (f64, f64) {
    if let Some((y0, y1)) = lowered_unary_eval_pair(op, x0, x1, mode) {
        (y0, y1)
    } else {
        (
            lowered_unary_eval(op, x0, mode),
            lowered_unary_eval(op, x1, mode),
        )
    }
}

pub(crate) fn lowered_expr_eval_pair(
    expr: &LoweredExpr,
    views: &[BufferView],
    idx0: usize,
    idx1: usize,
    mode: SimdMathMode,
) -> Option<(f64, f64)> {
    match expr {
        LoweredExpr::Const(v) => Some((*v, *v)),
        LoweredExpr::Input(input_idx) => {
            let view = views.get(*input_idx)?;
            Some((unsafe { read_buffer_f64(view, idx0) }, unsafe {
                read_buffer_f64(view, idx1)
            }))
        }
        LoweredExpr::Add(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 + b0, a1 + b1))
        }
        LoweredExpr::Sub(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 - b0, a1 - b1))
        }
        LoweredExpr::Mul(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 * b0, a1 * b1))
        }
        LoweredExpr::Div(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 / b0, a1 / b1))
        }
        LoweredExpr::Mod(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 % b0, a1 % b1))
        }
        LoweredExpr::Eq(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 == b0 { 1.0 } else { 0.0 },
                if a1 == b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Ne(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != b0 { 1.0 } else { 0.0 },
                if a1 != b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Lt(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 < b0 { 1.0 } else { 0.0 },
                if a1 < b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Gt(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 > b0 { 1.0 } else { 0.0 },
                if a1 > b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Le(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 <= b0 { 1.0 } else { 0.0 },
                if a1 <= b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Ge(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 >= b0 { 1.0 } else { 0.0 },
                if a1 >= b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::And(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != 0.0 && b0 != 0.0 { 1.0 } else { 0.0 },
                if a1 != 0.0 && b1 != 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Or(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != 0.0 || b0 != 0.0 { 1.0 } else { 0.0 },
                if a1 != 0.0 || b1 != 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Neg(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((-x0, -x1))
        }
        LoweredExpr::Not(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((
                if x0 == 0.0 { 1.0 } else { 0.0 },
                if x1 == 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Abs(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((x0.abs(), x1.abs()))
        }
        LoweredExpr::Sin(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Sin,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Cos(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Cos,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Tan(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Tan,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Exp(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Exp,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Log(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Log,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Sqrt(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Sqrt,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            let (c0, c1) = lowered_expr_eval_pair(cond, views, idx0, idx1, mode)?;
            let c0_true = c0 != 0.0;
            let c1_true = c1 != 0.0;
            match (c0_true, c1_true) {
                (true, true) => lowered_expr_eval_pair(then_expr, views, idx0, idx1, mode),
                (false, false) => lowered_expr_eval_pair(else_expr, views, idx0, idx1, mode),
                _ => {
                    let v0 = if c0_true {
                        lowered_expr_eval(then_expr, views, idx0, mode)?
                    } else {
                        lowered_expr_eval(else_expr, views, idx0, mode)?
                    };
                    let v1 = if c1_true {
                        lowered_expr_eval(then_expr, views, idx1, mode)?
                    } else {
                        lowered_expr_eval(else_expr, views, idx1, mode)?
                    };
                    Some((v0, v1))
                }
            }
        }
    }
}
