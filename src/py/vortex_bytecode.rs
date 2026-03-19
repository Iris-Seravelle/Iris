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
) -> Vec<Instruction> {
    if original.is_empty() {
        return original.to_vec();
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

    let _ = source_old_idx;
    let _ = old_to_new;

    out
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

    let bytes = locals.get_item("__iris_probe_bytes").unwrap().downcast::<PyList>()?;
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
        .get_item("__iris_probe_bytes")
        .unwrap()
        .downcast::<PyList>()?;
    bytes.extract()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instrument_inserts_entry_and_backedge_sites() {
        let meta = OpcodeMeta {
            extended_arg: 144,
            hasjabs: [113u16].into_iter().collect(),
            hasjrel: HashSet::new(),
            backward_relative: HashSet::new(),
        };

        let original = vec![
            Instruction { op: 9, arg: 0 },
            Instruction { op: 113, arg: 0 },
        ];
        let probe = vec![Instruction { op: 9, arg: 0 }];

        let patched = instrument_with_probe(&original, &probe, &meta);
        assert!(patched.len() > original.len());
    }
}
