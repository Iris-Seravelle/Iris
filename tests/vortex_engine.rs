#![cfg(feature = "vortex")]

use iris::vortex::{
    VortexEngine, VortexGhostPolicy, VortexInstruction, VortexSuspend, VortexVioCall,
};
use std::collections::HashMap;

#[test]
fn vortex_engine_new_is_enabled() {
    let engine = VortexEngine::new();
    assert!(engine.is_enabled());
}

#[test]
fn vortex_engine_preemption_and_transactions() {
    let mut engine = VortexEngine::new();

    assert!(engine.check_preemption(100));
    assert_eq!(engine.transmuter.instruction_budget, 924);

    engine.start_transaction(42);
    assert!(engine.commit_transaction());
    assert!(!engine.abort_transaction());

    engine.start_transaction(43);
    assert!(engine.abort_transaction());
    assert!(!engine.commit_transaction());

    engine.detach_stalled_thread();
    assert_eq!(engine.rescue_pool.active_count, 1);
    engine.reclaim_thread();
    assert_eq!(engine.rescue_pool.active_count, 0);
}

#[test]
fn vortex_engine_resume_after_suspend() {
    let mut engine = VortexEngine::new();
    engine.set_budget(2);

    let code = vec![
        VortexInstruction::LoadFast(0),
        VortexInstruction::BinaryOp(0),
        VortexInstruction::StoreFast(0),
        VortexInstruction::ReturnValue,
    ];

    engine.load_code(code);

    let first_run = engine.run();
    assert_eq!(first_run, Err(VortexSuspend));
    assert!(engine.current_code.is_some());
    assert!(engine.context.is_some());

    engine.replenish_budget(10);
    let second_run = engine.run();
    assert_eq!(second_run, Ok(()));
    assert!(engine.current_code.is_none());
    assert!(engine.context.is_none());
}

#[test]
fn vortex_engine_transaction_checkpoint_and_vio() {
    let mut engine = VortexEngine::new();
    let mut locals = HashMap::new();
    locals.insert("foo".to_string(), b"bar".to_vec());

    engine.start_transaction_with_checkpoint(91, locals);
    assert!(engine.stage_transaction_vio("send_email".to_string(), b"a".to_vec()));
    assert!(engine.stage_transaction_vio("write_db".to_string(), b"b".to_vec()));
    assert_eq!(engine.transaction_staged_vio_len(), 2);

    assert!(engine.commit_transaction());
    assert_eq!(engine.transaction_staged_vio_len(), 0);
    assert_eq!(engine.transaction_committed_vio_len(), 2);

    let committed = engine.take_committed_transaction_vio();
    assert_eq!(committed.len(), 2);
    assert_eq!(engine.transaction_committed_vio_len(), 0);
}

#[test]
fn vortex_engine_abort_clears_staged_vio() {
    let mut engine = VortexEngine::new();
    engine.start_transaction(92);
    assert!(engine.stage_transaction_vio("x".to_string(), vec![1]));
    assert!(engine.stage_transaction_vio("y".to_string(), vec![2]));
    assert_eq!(engine.transaction_staged_vio_len(), 2);

    assert!(engine.abort_transaction());
    assert_eq!(engine.transaction_staged_vio_len(), 0);
    assert_eq!(engine.transaction_committed_vio_len(), 0);
}

#[test]
fn vortex_engine_stage_swap_applies_when_idle() {
    let mut engine = VortexEngine::new();
    engine.stage_code_swap(vec![VortexInstruction::ReturnValue]);

    assert!(engine.try_apply_staged_swap());
    assert!(engine.pending_code_swap.is_none());
    assert!(engine.current_code.is_some());
    assert!(engine.context.is_some());
}

#[test]
fn vortex_engine_stage_swap_waits_for_quiescence() {
    let mut engine = VortexEngine::new();
    engine.set_budget(2);

    engine.load_code(vec![
        VortexInstruction::LoadFast(0),
        VortexInstruction::BinaryOp(0),
        VortexInstruction::ReturnValue,
    ]);
    assert_eq!(engine.run(), Err(VortexSuspend));

    engine.stage_code_swap(vec![VortexInstruction::ReturnValue]);
    assert!(!engine.try_apply_staged_swap());
    assert!(engine.pending_code_swap.is_some());
}

#[test]
fn vortex_engine_stage_swap_applies_after_completion() {
    let mut engine = VortexEngine::new();
    engine.set_budget(1);

    engine.load_code(vec![
        VortexInstruction::LoadFast(0),
        VortexInstruction::ReturnValue,
    ]);
    engine.stage_code_swap(vec![VortexInstruction::ReturnValue]);

    assert_eq!(engine.run(), Err(VortexSuspend));
    engine.replenish_budget(10);

    assert_eq!(engine.run(), Ok(()));
    assert!(engine.pending_code_swap.is_none());
    assert!(engine.current_code.is_some());
    assert!(engine.context.is_some());
}

#[test]
fn vortex_engine_resolve_primary_ghost_race_and_replay() {
    let mut engine = VortexEngine::new();

    let mut primary_locals = HashMap::new();
    primary_locals.insert("count".to_string(), vec![1]);
    engine.start_transaction_with_checkpoint(1000, primary_locals);
    assert!(engine.stage_transaction_vio("io_primary".to_string(), b"p".to_vec()));

    let mut ghost_locals = HashMap::new();
    ghost_locals.insert("count".to_string(), vec![2]);
    engine.start_ghost_transaction_with_checkpoint(2000, ghost_locals);
    assert!(engine.stage_ghost_transaction_vio(2000, "io_ghost_a".to_string(), b"a".to_vec()));
    assert!(engine.stage_ghost_transaction_vio(2000, "io_ghost_b".to_string(), b"b".to_vec()));

    let resolution = engine
        .resolve_primary_ghost_race(2000, 2000, VortexGhostPolicy::FirstSafePointWins)
        .expect("resolution expected");
    assert_eq!(resolution.winner_id, 2000);
    assert_eq!(resolution.committed_vio.len(), 2);

    let applied = engine.replay_committed_vio_calls(&resolution.committed_vio, |_call| true);
    assert_eq!(applied, 2);
}

#[test]
fn vortex_engine_prefer_primary_policy_and_ghost_cleanup() {
    let mut engine = VortexEngine::new();

    engine.start_transaction_with_checkpoint(3000, HashMap::new());
    assert!(engine.stage_transaction_vio("io_primary".to_string(), b"p".to_vec()));

    engine.start_ghost_transaction_with_checkpoint(4000, HashMap::new());
    assert!(engine.stage_ghost_transaction_vio(4000, "io_ghost".to_string(), b"g".to_vec()));

    let resolution = engine
        .resolve_primary_ghost_race(4000, 4000, VortexGhostPolicy::PreferPrimary)
        .expect("resolution expected");
    assert_eq!(resolution.winner_id, 3000);
    assert_eq!(resolution.loser_id, 4000);
    assert_eq!(resolution.committed_vio.len(), 1);
    assert_eq!(resolution.committed_vio[0].op, "io_primary");

    let second = engine.resolve_primary_ghost_race(4000, 4000, VortexGhostPolicy::PreferPrimary);
    assert!(second.is_none());
}

#[test]
fn vortex_engine_replay_stops_on_executor_failure() {
    let engine = VortexEngine::new();
    let calls = vec![
        VortexVioCall {
            op: "a".to_string(),
            payload: vec![1],
        },
        VortexVioCall {
            op: "b".to_string(),
            payload: vec![2],
        },
    ];

    let mut seen = 0usize;
    let applied = engine.replay_committed_vio_calls(&calls, |_call| {
        seen += 1;
        false
    });

    assert_eq!(applied, 1);
    assert_eq!(seen, 1);
}
