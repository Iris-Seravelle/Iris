use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct OpcodeMeta {
    pub extended_arg: u8,
    pub hasjabs: HashSet<u16>,
    pub hasjrel: HashSet<u16>,
    pub backward_relative: HashSet<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub op: u8,
    pub arg: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    InvalidWordcodeShape,
    EmptyProbe,
    OversizedCode,
    InvalidJumpTarget,
    InvalidRelativeJump,
    InvalidCacheLayout,
    InvalidExceptionTable,
    StackDepthInvariant,
}

#[derive(Debug, Clone)]
pub struct QuickeningSupport {
    pub cache_opcode: Option<u8>,
    pub inline_cache_entries: Vec<u16>,
}

const MAX_WORDCODE_BYTES: usize = 4 * 1024 * 1024;

pub fn verify_wordcode_bytes(raw: &[u8]) -> Result<(), VerifyError> {
    if raw.is_empty() || raw.len() % 2 != 0 {
        return Err(VerifyError::InvalidWordcodeShape);
    }
    if raw.len() > MAX_WORDCODE_BYTES {
        return Err(VerifyError::OversizedCode);
    }
    Ok(())
}

pub fn opcode_meta(py: Python) -> PyResult<OpcodeMeta> {
    let meta = py.eval(
        r#"
(lambda dis: (
    dis.opmap["EXTENDED_ARG"],
    list(dis.hasjabs),
    list(dis.hasjrel),
    [op for name, op in dis.opmap.items() if "BACKWARD" in name],
))(__import__("dis"))
"#,
        None,
        None,
    )?;

    let extended_arg: u8 = meta.get_item(0)?.extract()?;
    let hasjabs: Vec<u16> = meta.get_item(1)?.extract()?;
    let hasjrel: Vec<u16> = meta.get_item(2)?.extract()?;
    let backward_relative: Vec<u16> = meta.get_item(3)?.extract()?;

    Ok(OpcodeMeta {
        extended_arg,
        hasjabs: hasjabs.into_iter().collect(),
        hasjrel: hasjrel.into_iter().collect(),
        backward_relative: backward_relative.into_iter().collect(),
    })
}

pub fn quickening_support(py: Python) -> PyResult<QuickeningSupport> {
    let data = py.eval(
        r#"
(lambda dis: (
    dis.opmap.get("CACHE", -1),
    list(getattr(dis, "_inline_cache_entries", [])),
))(__import__("dis"))
"#,
        None,
        None,
    )?;

    let cache_raw: i16 = data.get_item(0)?.extract()?;
    let entries: Vec<u16> = data.get_item(1)?.extract()?;
    let cache_opcode = if cache_raw >= 0 {
        Some(cache_raw as u8)
    } else {
        None
    };

    Ok(QuickeningSupport {
        cache_opcode,
        inline_cache_entries: entries,
    })
}

pub fn decode_wordcode(raw: &[u8], extended_arg: u8) -> Vec<Instruction> {
    let mut out = Vec::new();
    let mut ext: u32 = 0;
    let mut i = 0usize;

    while i + 1 < raw.len() {
        let op = raw[i];
        let arg = raw[i + 1] as u32;
        if op == extended_arg {
            ext = (ext << 8) | arg;
            i += 2;
            continue;
        }

        let full_arg = (ext << 8) | arg;
        out.push(Instruction { op, arg: full_arg });
        ext = 0;
        i += 2;
    }

    out
}

pub fn encode_wordcode(instructions: &[Instruction], extended_arg: u8) -> Vec<u8> {
    let mut out = Vec::new();

    for ins in instructions {
        let mut high = ins.arg >> 8;
        let mut ext = Vec::new();
        while high > 0 {
            ext.push((high & 0xFF) as u8);
            high >>= 8;
        }

        for b in ext.iter().rev() {
            out.push(extended_arg);
            out.push(*b);
        }

        out.push(ins.op);
        out.push((ins.arg & 0xFF) as u8);
    }

    out
}

fn jump_target(idx: usize, ins: &Instruction, meta: &OpcodeMeta, len: usize) -> Option<usize> {
    if meta.hasjabs.contains(&(ins.op as u16)) {
        let t = ins.arg as usize;
        return (t < len).then_some(t);
    }

    if meta.hasjrel.contains(&(ins.op as u16)) {
        if meta.backward_relative.contains(&(ins.op as u16)) {
            let base = idx + 1;
            if (ins.arg as usize) > base {
                return None;
            }
            return Some(base - ins.arg as usize);
        }

        let t = idx + 1 + ins.arg as usize;
        return (t < len).then_some(t);
    }

    None
}

pub fn instrument_with_probe(
    original: &[Instruction],
    probe: &[Instruction],
    meta: &OpcodeMeta,
) -> Result<Vec<Instruction>, VerifyError> {
    if probe.is_empty() {
        return Err(VerifyError::EmptyProbe);
    }

    if original.is_empty() {
        return Ok(original.to_vec());
    }

    let mut check_sites: HashSet<usize> = HashSet::new();
    check_sites.insert(0);

    for (idx, ins) in original.iter().enumerate() {
        if let Some(target) = jump_target(idx, ins, meta, original.len()) {
            if target <= idx {
                check_sites.insert(target);
            }
        }
    }

    let mut out = Vec::new();
    let mut old_to_new = vec![0usize; original.len()];
    let mut source_old_idx = Vec::new();

    for (idx, ins) in original.iter().enumerate() {
        if check_sites.contains(&idx) {
            for p in probe {
                out.push(p.clone());
                source_old_idx.push(None);
            }
        }
        old_to_new[idx] = out.len();
        out.push(ins.clone());
        source_old_idx.push(Some(idx));
    }

    for (new_idx, src) in source_old_idx.iter().enumerate() {
        let Some(old_idx) = src else {
            continue;
        };

        let current = out[new_idx].clone();
        let Some(old_target) = jump_target(*old_idx, &current, meta, original.len()) else {
            continue;
        };
        let new_target = old_to_new[old_target];

        if meta.hasjabs.contains(&(current.op as u16)) {
            out[new_idx].arg = new_target as u32;
            continue;
        }

        if meta.hasjrel.contains(&(current.op as u16)) {
            if meta.backward_relative.contains(&(current.op as u16)) {
                if new_target > new_idx + 1 {
                    return Err(VerifyError::InvalidRelativeJump);
                }
                out[new_idx].arg = (new_idx + 1 - new_target) as u32;
            } else {
                if new_target < new_idx + 1 {
                    return Err(VerifyError::InvalidRelativeJump);
                }
                out[new_idx].arg = (new_target - (new_idx + 1)) as u32;
            }
        }
    }

    verify_instructions(&out, meta)?;
    Ok(out)
}

pub fn verify_instructions(code: &[Instruction], meta: &OpcodeMeta) -> Result<(), VerifyError> {
    if code.is_empty() {
        return Ok(());
    }

    for (idx, ins) in code.iter().enumerate() {
        if meta.hasjabs.contains(&(ins.op as u16)) {
            if (ins.arg as usize) >= code.len() {
                return Err(VerifyError::InvalidJumpTarget);
            }
            continue;
        }

        if !meta.hasjrel.contains(&(ins.op as u16)) {
            continue;
        }

        if meta.backward_relative.contains(&(ins.op as u16)) {
            let base = idx + 1;
            if (ins.arg as usize) > base {
                return Err(VerifyError::InvalidRelativeJump);
            }
            let t = base - ins.arg as usize;
            if t >= code.len() {
                return Err(VerifyError::InvalidJumpTarget);
            }
        } else {
            let t = idx + 1 + ins.arg as usize;
            if t >= code.len() {
                return Err(VerifyError::InvalidJumpTarget);
            }
        }
    }

    Ok(())
}

pub fn verify_cache_layout(
    code: &[Instruction],
    quickening: &QuickeningSupport,
) -> Result<(), VerifyError> {
    let Some(cache_opcode) = quickening.cache_opcode else {
        return Ok(());
    };

    if quickening.inline_cache_entries.is_empty() {
        return Ok(());
    }

    let mut i = 0usize;
    while i < code.len() {
        let op = code[i].op as usize;

        if code[i].op == cache_opcode {
            i += 1;
            continue;
        }

        let expected_caches = quickening
            .inline_cache_entries
            .get(op)
            .copied()
            .unwrap_or(0) as usize;
        for j in 0..expected_caches {
            let next = i + 1 + j;
            if next >= code.len() || code[next].op != cache_opcode {
                return Err(VerifyError::InvalidCacheLayout);
            }
        }

        i += 1 + expected_caches;
    }

    Ok(())
}

pub fn evaluate_rewrite_compatibility(
    raw: &[u8],
    extended_arg: u8,
    quickening: &QuickeningSupport,
) -> Result<(), &'static str> {
    match verify_wordcode_bytes(raw) {
        Ok(()) => {}
        Err(VerifyError::InvalidWordcodeShape) => return Err("invalid_wordcode_shape"),
        Err(VerifyError::OversizedCode) => return Err("oversized_wordcode"),
        Err(_) => return Err("invalid_wordcode"),
    }

    if quickening.cache_opcode.is_some() && quickening.inline_cache_entries.len() < 256 {
        return Err("inline_cache_entries_incomplete");
    }

    let original = decode_wordcode(raw, extended_arg);
    if verify_cache_layout(&original, quickening).is_err() {
        return Err("original_cache_layout_invalid");
    }

    Ok(())
}

pub fn validate_probe_compatibility(
    probe: &[Instruction],
    quickening: &QuickeningSupport,
) -> Result<(), &'static str> {
    if probe.is_empty() {
        return Err("empty_probe");
    }

    if verify_cache_layout(probe, quickening).is_err() {
        return Err("probe_cache_layout_invalid");
    }

    Ok(())
}

pub fn read_exception_entries(
    py: Python,
    code: &PyAny,
) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    let locals = PyDict::new(py);
    locals.set_item("code_obj", code)?;
    py.run(
        r#"
import dis
bc = dis.Bytecode(code_obj)
entries = getattr(bc, "exception_entries", ())
__iris_exc_entries = [
    (
        int(e.start),
        int(e.end),
        int(e.depth),
        int(getattr(e, "target", e.start)),
    )
    for e in entries
]
"#,
        None,
        Some(locals),
    )?;

    let entries = locals
        .get_item("__iris_exc_entries")?
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("vortex/exception-entries: missing result")
        })?
        .downcast::<PyList>()?;
    entries.extract()
}

pub fn verify_exception_table_invariants(
    entries: &[(usize, usize, usize, usize)],
    code_units: usize,
    stack_size: usize,
) -> Result<(), VerifyError> {
    let mut seen = HashSet::with_capacity(entries.len());
    let mut prev: Option<(usize, usize, usize, usize)> = None;

    for (start, end, depth, target) in entries {
        if *start >= *end || *end > code_units {
            return Err(VerifyError::InvalidExceptionTable);
        }
        if *depth > stack_size {
            return Err(VerifyError::StackDepthInvariant);
        }
        if *target >= code_units {
            return Err(VerifyError::InvalidExceptionTable);
        }

        let current = (*start, *end, *depth, *target);
        if let Some(p) = prev {
            if current < p {
                return Err(VerifyError::InvalidExceptionTable);
            }
        }
        if !seen.insert(current) {
            return Err(VerifyError::InvalidExceptionTable);
        }
        prev = Some(current);
    }
    Ok(())
}

pub fn verify_exception_handler_targets(
    entries: &[(usize, usize, usize, usize)],
    code: &[Instruction],
    quickening: &QuickeningSupport,
) -> Result<(), VerifyError> {
    let Some(cache_opcode) = quickening.cache_opcode else {
        return Ok(());
    };

    for (_, _, _, target) in entries {
        if *target >= code.len() {
            return Err(VerifyError::InvalidExceptionTable);
        }
        if code[*target].op == cache_opcode {
            return Err(VerifyError::InvalidExceptionTable);
        }
    }

    Ok(())
}

pub fn apply_isolation_transform(
    code: &[Instruction],
    py: Python,
    disallowed_ops: Option<&std::collections::HashSet<u8>>,
) -> PyResult<Vec<Instruction>> {
    let dis = py.import("dis")?;
    let store_attr: u8 = dis.getattr("opmap")?.get_item("STORE_ATTR")?.extract()?;

    if let Some(disallowed) = disallowed_ops {
        for ins in code {
            if disallowed.contains(&ins.op) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "isolation disallowed opcode encountered",
                ));
            }
        }
    }

    // In strict isolation mode, global/name stores are allowed because transmuted
    // function globals are detached from module state. Attribute stores remain unsafe
    // (object side effects can escape), so they are rejected.
    for ins in code {
        if ins.op == store_attr {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "isolation unsafe STORE_ATTR opcode encountered",
            ));
        }
    }

    Ok(code.to_vec())
}

pub fn verify_stacksize_minimum(stack_size: usize) -> Result<(), VerifyError> {
    // Probe executes a callable check and requires temporary stack headroom.
    if stack_size < 2 {
        return Err(VerifyError::StackDepthInvariant);
    }
    Ok(())
}

pub fn probe_instructions(py: Python, extended_arg: u8) -> PyResult<Vec<Instruction>> {
    let locals = PyDict::new(py);
    py.run(
        r#"
import dis

def __iris_probe():
    _vortex_check()

ins = list(dis.get_instructions(__iris_probe, show_caches=True))
start = next(i.offset for i in ins if i.opname == "LOAD_GLOBAL")
end = next(i.offset for i in ins if i.opname == "POP_TOP")
__iris_probe_bytes = list(__iris_probe.__code__.co_code[start:end+2])
"#,
        None,
        Some(locals),
    )?;

    let bytes = locals
        .get_item("__iris_probe_bytes")?
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("vortex/probe-bytes: missing result")
        })?
        .downcast::<PyList>()?;
    let raw: Vec<u8> = bytes.extract()?;
    Ok(decode_wordcode(&raw, extended_arg))
}

pub fn probe_raw_bytes(py: Python) -> PyResult<Vec<u8>> {
    let locals = PyDict::new(py);
    py.run(
        r#"
import dis

def __iris_probe():
    _vortex_check()

ins = list(dis.get_instructions(__iris_probe, show_caches=True))
start = next(i.offset for i in ins if i.opname == "LOAD_GLOBAL")
end = next(i.offset for i in ins if i.opname == "POP_TOP")
__iris_probe_bytes = list(__iris_probe.__code__.co_code[start:end+2])
"#,
        None,
        Some(locals),
    )?;

    let bytes = locals
        .get_item("__iris_probe_bytes")?
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("vortex/probe-bytes: missing result")
        })?
        .downcast::<PyList>()?;
    bytes.extract()
}
