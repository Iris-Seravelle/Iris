#![cfg(feature = "vortex")]

use iris::vortex::vortex_bytecode::*;
use std::collections::HashSet;

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

    let patched =
        instrument_with_probe(&original, &probe, &meta).expect("instrumentation should be valid");
    assert!(patched.len() > original.len());
}

#[test]
fn verify_wordcode_bytes_rejects_invalid_shape() {
    assert_eq!(
        verify_wordcode_bytes(&[]),
        Err(VerifyError::InvalidWordcodeShape)
    );
    assert_eq!(
        verify_wordcode_bytes(&[1]),
        Err(VerifyError::InvalidWordcodeShape)
    );
}

#[test]
fn verify_instructions_rejects_bad_abs_jump() {
    let meta = OpcodeMeta {
        extended_arg: 144,
        hasjabs: [113u16].into_iter().collect(),
        hasjrel: HashSet::new(),
        backward_relative: HashSet::new(),
    };

    let code = vec![Instruction { op: 113, arg: 99 }];
    assert_eq!(
        verify_instructions(&code, &meta),
        Err(VerifyError::InvalidJumpTarget)
    );
}

#[test]
fn verify_instructions_rejects_bad_backward_rel_jump() {
    let meta = OpcodeMeta {
        extended_arg: 144,
        hasjabs: HashSet::new(),
        hasjrel: [200u16].into_iter().collect(),
        backward_relative: [200u16].into_iter().collect(),
    };

    let code = vec![Instruction { op: 200, arg: 2 }];
    assert_eq!(
        verify_instructions(&code, &meta),
        Err(VerifyError::InvalidRelativeJump)
    );
}

#[test]
fn verify_cache_layout_rejects_missing_cache_slots() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: {
            let mut v = vec![0u16; 256];
            v[10] = 2;
            v
        },
    };

    let code = vec![
        Instruction { op: 10, arg: 0 },
        Instruction { op: 0, arg: 0 },
    ];
    assert_eq!(
        verify_cache_layout(&code, &quick),
        Err(VerifyError::InvalidCacheLayout)
    );
}

#[test]
fn verify_cache_layout_accepts_expected_cache_slots() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: {
            let mut v = vec![0u16; 256];
            v[10] = 2;
            v
        },
    };

    let code = vec![
        Instruction { op: 10, arg: 0 },
        Instruction { op: 0, arg: 0 },
        Instruction { op: 0, arg: 0 },
        Instruction { op: 5, arg: 0 },
    ];
    assert_eq!(verify_cache_layout(&code, &quick), Ok(()));
}

#[test]
fn evaluate_rewrite_compatibility_rejects_incomplete_cache_table() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: vec![0u16; 8],
    };

    let raw = vec![5u8, 0u8];
    assert_eq!(
        evaluate_rewrite_compatibility(&raw, 144, &quick),
        Err("inline_cache_entries_incomplete")
    );
}

#[test]
fn evaluate_rewrite_compatibility_rejects_invalid_raw_shape() {
    let quick = QuickeningSupport {
        cache_opcode: None,
        inline_cache_entries: vec![],
    };

    let raw = vec![1u8];
    assert_eq!(
        evaluate_rewrite_compatibility(&raw, 144, &quick),
        Err("invalid_wordcode_shape")
    );
}

#[test]
fn evaluate_rewrite_compatibility_accepts_minimal_non_quickened() {
    let quick = QuickeningSupport {
        cache_opcode: None,
        inline_cache_entries: vec![],
    };

    let raw = vec![5u8, 0u8, 6u8, 0u8];
    assert_eq!(evaluate_rewrite_compatibility(&raw, 144, &quick), Ok(()));
}

#[test]
fn evaluate_rewrite_compatibility_rejects_invalid_original_cache_layout() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: {
            let mut v = vec![0u16; 256];
            v[10] = 1;
            v
        },
    };

    let raw = vec![10u8, 0u8, 5u8, 0u8];
    assert_eq!(
        evaluate_rewrite_compatibility(&raw, 144, &quick),
        Err("original_cache_layout_invalid")
    );
}

#[test]
fn validate_probe_compatibility_rejects_empty_probe() {
    let quick = QuickeningSupport {
        cache_opcode: None,
        inline_cache_entries: vec![],
    };
    assert_eq!(
        validate_probe_compatibility(&[], &quick),
        Err("empty_probe")
    );
}

#[test]
fn verify_exception_table_invariants_rejects_out_of_range_entry() {
    let entries = vec![(0usize, 5usize, 1usize, 0usize)];
    assert_eq!(
        verify_exception_table_invariants(&entries, 4, 8),
        Err(VerifyError::InvalidExceptionTable)
    );
}

#[test]
fn verify_exception_table_invariants_rejects_depth_over_stack() {
    let entries = vec![(0usize, 1usize, 9usize, 0usize)];
    assert_eq!(
        verify_exception_table_invariants(&entries, 4, 8),
        Err(VerifyError::StackDepthInvariant)
    );
}

#[test]
fn verify_exception_table_invariants_rejects_unsorted_ranges() {
    let entries = vec![
        (2usize, 3usize, 0usize, 0usize),
        (0usize, 1usize, 0usize, 0usize),
    ];
    assert_eq!(
        verify_exception_table_invariants(&entries, 4, 8),
        Err(VerifyError::InvalidExceptionTable)
    );
}

#[test]
fn verify_exception_table_invariants_rejects_duplicate_entries() {
    let entries = vec![
        (0usize, 1usize, 0usize, 2usize),
        (0usize, 1usize, 0usize, 2usize),
    ];
    assert_eq!(
        verify_exception_table_invariants(&entries, 4, 8),
        Err(VerifyError::InvalidExceptionTable)
    );
}

#[test]
fn verify_exception_table_invariants_accepts_sorted_unique_entries() {
    let entries = vec![
        (0usize, 1usize, 0usize, 2usize),
        (1usize, 3usize, 1usize, 3usize),
    ];
    assert_eq!(verify_exception_table_invariants(&entries, 4, 8), Ok(()));
}

#[test]
fn verify_exception_table_invariants_rejects_handler_target_out_of_range() {
    let entries = vec![(0usize, 1usize, 0usize, 9usize)];
    assert_eq!(
        verify_exception_table_invariants(&entries, 4, 8),
        Err(VerifyError::InvalidExceptionTable)
    );
}

#[test]
fn verify_exception_handler_targets_rejects_cache_opcode_target() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: vec![0u16; 256],
    };
    let code = vec![
        Instruction { op: 10, arg: 0 },
        Instruction { op: 0, arg: 0 },
    ];
    let entries = vec![(0usize, 1usize, 0usize, 1usize)];

    assert_eq!(
        verify_exception_handler_targets(&entries, &code, &quick),
        Err(VerifyError::InvalidExceptionTable)
    );
}

#[test]
fn verify_exception_handler_targets_accepts_non_cache_target() {
    let quick = QuickeningSupport {
        cache_opcode: Some(0),
        inline_cache_entries: vec![0u16; 256],
    };
    let code = vec![
        Instruction { op: 10, arg: 0 },
        Instruction { op: 0, arg: 0 },
        Instruction { op: 5, arg: 0 },
    ];
    let entries = vec![(0usize, 1usize, 0usize, 2usize)];

    assert_eq!(
        verify_exception_handler_targets(&entries, &code, &quick),
        Ok(())
    );
}

#[test]
fn verify_stacksize_minimum_rejects_tiny_stack() {
    assert_eq!(
        verify_stacksize_minimum(1),
        Err(VerifyError::StackDepthInvariant)
    );
    assert_eq!(verify_stacksize_minimum(2), Ok(()));
}
