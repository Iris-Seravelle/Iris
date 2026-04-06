#![cfg(feature = "vortex")]

use iris::vortex::{VortexGhostPolicy, VortexTransaction};
use std::collections::HashMap;

#[test]
fn vortex_transaction_commit_abort() {
    let mut trx = VortexTransaction::new(1);
    assert!(trx.commit());
    assert!(!trx.abort());
    assert!(!trx.commit());

    let mut trx2 = VortexTransaction::new(2);
    assert!(trx2.abort());
    assert!(!trx2.commit());
}

#[test]
fn vortex_transaction_checkpoint_and_vio_commit() {
    let mut trx = VortexTransaction::new(10);
    let mut locals = HashMap::new();
    locals.insert("counter".to_string(), vec![1, 2, 3]);
    trx.checkpoint_locals(locals.clone());

    assert_eq!(trx.local_checkpoint.get("counter"), Some(&vec![1, 2, 3]));

    assert!(trx.stage_vio("send_email".to_string(), b"payload-a".to_vec()));
    assert!(trx.stage_vio("write_db".to_string(), b"payload-b".to_vec()));
    assert_eq!(trx.staged_vio_len(), 2);
    assert_eq!(trx.committed_vio_len(), 0);

    assert!(trx.commit());
    assert_eq!(trx.staged_vio_len(), 0);
    assert_eq!(trx.committed_vio_len(), 2);

    let drained = trx.drain_committed_vio();
    assert_eq!(drained.len(), 2);
    assert_eq!(trx.committed_vio_len(), 0);
}

#[test]
fn vortex_transaction_abort_discards_staged_vio() {
    let mut trx = VortexTransaction::new(11);
    assert!(trx.stage_vio("io_a".to_string(), vec![9]));
    assert!(trx.stage_vio("io_b".to_string(), vec![8]));
    assert_eq!(trx.staged_vio_len(), 2);

    assert!(trx.abort());
    assert_eq!(trx.staged_vio_len(), 0);
    assert_eq!(trx.committed_vio_len(), 0);
    assert!(!trx.stage_vio("io_c".to_string(), vec![7]));
}

#[test]
fn vortex_transaction_resolve_ghost_race_first_wins() {
    let mut primary = VortexTransaction::new(100);
    let mut ghost = VortexTransaction::new(200);

    assert!(primary.stage_vio("io_primary".to_string(), b"p".to_vec()));
    assert!(ghost.stage_vio("io_ghost".to_string(), b"g".to_vec()));

    let result = VortexTransaction::resolve_ghost_race(
        &mut primary,
        &mut ghost,
        200,
        VortexGhostPolicy::FirstSafePointWins,
    )
    .expect("resolution should succeed");

    assert_eq!(result.winner_id, 200);
    assert_eq!(result.loser_id, 100);
    assert_eq!(result.committed_vio.len(), 1);
    assert!(primary.aborted);
    assert!(ghost.committed);
}

#[test]
fn vortex_transaction_resolve_ghost_race_prefer_primary() {
    let mut primary = VortexTransaction::new(101);
    let mut ghost = VortexTransaction::new(201);

    assert!(primary.stage_vio("io_primary".to_string(), b"p".to_vec()));
    assert!(ghost.stage_vio("io_ghost".to_string(), b"g".to_vec()));

    let result = VortexTransaction::resolve_ghost_race(
        &mut primary,
        &mut ghost,
        201,
        VortexGhostPolicy::PreferPrimary,
    )
    .expect("resolution should succeed");

    assert_eq!(result.winner_id, 101);
    assert_eq!(result.loser_id, 201);
    assert_eq!(result.committed_vio.len(), 1);
    assert!(primary.committed);
    assert!(ghost.aborted);
}
