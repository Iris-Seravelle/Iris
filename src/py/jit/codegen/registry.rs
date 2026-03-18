// src/py/jit/codegen/registry.rs
//! Registry & invocation helpers for compiled JIT entries.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::{Lazy, OnceCell};

use super::JitEntry;

static JIT_FUNC_COUNTER: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

static JIT_REGISTRY: OnceCell<std::sync::Mutex<HashMap<usize, JitEntry>>> =
    OnceCell::new();

static NAMED_JIT_REGISTRY: OnceCell<std::sync::Mutex<HashMap<String, JitEntry>>> =
    OnceCell::new();

pub fn next_jit_func_id() -> usize {
    JIT_FUNC_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub fn register_jit(func_key: usize, entry: JitEntry) {
    let map = JIT_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(func_key, entry);
}

pub fn lookup_jit(func_key: usize) -> Option<JitEntry> {
    JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(&func_key).cloned())
}

pub fn register_named_jit(name: &str, entry: JitEntry) {
    let map = NAMED_JIT_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(name.to_string(), entry);
}

pub fn lookup_named_jit(name: &str) -> Option<JitEntry> {
    NAMED_JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(name).cloned())
}

fn invoke_named_entry(func_ptr: i64, args: &[f64]) -> f64 {
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(func_ptr as usize) };
    f(args.as_ptr())
}

macro_rules! make_invoke {
    ($name:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(func_ptr: i64) -> f64 {
            let args: [f64; 0] = [];
            invoke_named_entry(func_ptr, &args)
        }
    };
    ($name:ident, $($arg:ident),+) => {
        #[no_mangle]
        pub extern "C" fn $name(func_ptr: i64, $($arg: f64),+) -> f64 {
            let args = [$($arg),+];
            invoke_named_entry(func_ptr, &args)
        }
    };
}

make_invoke!(iris_jit_invoke_0);
make_invoke!(iris_jit_invoke_1, a0);
make_invoke!(iris_jit_invoke_2, a0, a1);
make_invoke!(iris_jit_invoke_3, a0, a1, a2);
make_invoke!(iris_jit_invoke_4, a0, a1, a2, a3);
make_invoke!(iris_jit_invoke_5, a0, a1, a2, a3, a4);
make_invoke!(iris_jit_invoke_6, a0, a1, a2, a3, a4, a5);
make_invoke!(iris_jit_invoke_7, a0, a1, a2, a3, a4, a5, a6);
make_invoke!(iris_jit_invoke_8, a0, a1, a2, a3, a4, a5, a6, a7);
make_invoke!(iris_jit_invoke_9, a0, a1, a2, a3, a4, a5, a6, a7, a8);
make_invoke!(
    iris_jit_invoke_10,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9
);
make_invoke!(
    iris_jit_invoke_11,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10
);
make_invoke!(
    iris_jit_invoke_12,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11
);
make_invoke!(
    iris_jit_invoke_13,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12
);
make_invoke!(
    iris_jit_invoke_14,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13
);
make_invoke!(
    iris_jit_invoke_15,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14
);
make_invoke!(
    iris_jit_invoke_16,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15
);
