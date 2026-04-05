// src/vortex/transaction.rs
//! Experimental speculative transactional fiber ghosting stub.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexVioCall {
    pub op: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexGhostPolicy {
    FirstSafePointWins,
    PreferPrimary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexGhostResolution {
    pub winner_id: u64,
    pub loser_id: u64,
    pub committed_vio: Vec<VortexVioCall>,
}

#[derive(Debug, Clone)]
pub struct VortexTransaction {
    pub id: u64,
    pub committed: bool,
    pub aborted: bool,
    pub local_checkpoint: HashMap<String, Vec<u8>>,
    staged_vio: Vec<VortexVioCall>,
    committed_vio: Vec<VortexVioCall>,
}

impl VortexTransaction {
    pub fn new(id: u64) -> Self {
        VortexTransaction {
            id,
            committed: false,
            aborted: false,
            local_checkpoint: HashMap::new(),
            staged_vio: Vec::new(),
            committed_vio: Vec::new(),
        }
    }

    pub fn checkpoint_locals(&mut self, locals: HashMap<String, Vec<u8>>) {
        if self.committed || self.aborted {
            return;
        }
        self.local_checkpoint = locals;
    }

    pub fn stage_vio(&mut self, op: String, payload: Vec<u8>) -> bool {
        if self.committed || self.aborted {
            return false;
        }
        self.staged_vio.push(VortexVioCall { op, payload });
        true
    }

    pub fn staged_vio_len(&self) -> usize {
        self.staged_vio.len()
    }

    pub fn committed_vio_len(&self) -> usize {
        self.committed_vio.len()
    }

    pub fn drain_committed_vio(&mut self) -> Vec<VortexVioCall> {
        std::mem::take(&mut self.committed_vio)
    }

    pub fn resolve_ghost_race(
        primary: &mut VortexTransaction,
        ghost: &mut VortexTransaction,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let resolved_winner = match policy {
            VortexGhostPolicy::FirstSafePointWins => winner_id,
            VortexGhostPolicy::PreferPrimary => primary.id,
        };

        if resolved_winner == primary.id {
            if !primary.commit() {
                return None;
            }
            let _ = ghost.abort();
            return Some(VortexGhostResolution {
                winner_id: primary.id,
                loser_id: ghost.id,
                committed_vio: primary.drain_committed_vio(),
            });
        }

        if resolved_winner == ghost.id {
            if !ghost.commit() {
                return None;
            }
            let _ = primary.abort();
            return Some(VortexGhostResolution {
                winner_id: ghost.id,
                loser_id: primary.id,
                committed_vio: ghost.drain_committed_vio(),
            });
        }

        None
    }

    pub fn commit(&mut self) -> bool {
        if self.aborted || self.committed {
            return false;
        }
        self.committed_vio.extend(self.staged_vio.drain(..));
        self.committed = true;
        true
    }

    pub fn abort(&mut self) -> bool {
        if self.committed || self.aborted {
            return false;
        }
        self.staged_vio.clear();
        self.aborted = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
