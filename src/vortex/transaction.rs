// src/vortex/transaction.rs
//! Experimental speculative transactional fiber ghosting stub.

#[derive(Debug, Clone)]
pub struct VortexTransaction {
    pub id: u64,
    pub committed: bool,
    pub aborted: bool,
}

impl VortexTransaction {
    pub fn new(id: u64) -> Self {
        VortexTransaction {
            id,
            committed: false,
            aborted: false,
        }
    }

    pub fn commit(&mut self) -> bool {
        if self.aborted || self.committed {
            return false;
        }
        self.committed = true;
        true
    }

    pub fn abort(&mut self) -> bool {
        if self.committed || self.aborted {
            return false;
        }
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
}