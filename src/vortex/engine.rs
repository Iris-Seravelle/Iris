// src/vortex/engine.rs
//! Experimental Vortex runtime engine.

use std::collections::HashMap;

use crate::vortex::rescue_pool::RescuePool;
use crate::vortex::transaction::{
    VortexGhostPolicy, VortexGhostResolution, VortexTransaction, VortexVioCall,
};
use crate::vortex::transmuter::{
    VortexExecutionContext, VortexInstruction, VortexSuspend, VortexTransmuter,
};

#[derive(Debug, Clone)]
pub struct VortexEngine {
    pub enabled: bool,
    pub transmuter: VortexTransmuter,
    pub transaction: Option<VortexTransaction>,
    pub rescue_pool: RescuePool,
    pub budget: usize,
    pub current_code: Option<Vec<VortexInstruction>>,
    pub context: Option<VortexExecutionContext>,
    pub pending_code_swap: Option<Vec<VortexInstruction>>,
    ghost_transactions: HashMap<u64, VortexTransaction>,
}

impl VortexEngine {
    pub fn new() -> Self {
        VortexEngine {
            enabled: true,
            transmuter: VortexTransmuter::new(1024),
            transaction: None,
            rescue_pool: RescuePool::new(),
            budget: 1024,
            current_code: None,
            context: None,
            pending_code_swap: None,
            ghost_transactions: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn check_preemption(&mut self, instruction_cost: usize) -> bool {
        if !self.enabled {
            return false;
        }

        let should_continue = self.transmuter.inject_reduction_checks(instruction_cost);
        if !should_continue {
            self.enabled = false;
        }

        should_continue
    }

    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.transmuter.instruction_budget = budget;
        self.enabled = true;
    }

    /// Consume one reduction tick (Check) via transmuter instrumentation. Returns Err(VortexSuspend) if budget is exhausted.
    pub fn preempt_tick(&mut self) -> Result<(), VortexSuspend> {
        if !self.enabled {
            return Err(VortexSuspend);
        }

        // Execute only the injected reduction check opcode.
        let program = vec![
            VortexInstruction::IrisReductionCheck,
            VortexInstruction::ReturnValue,
        ];
        let mut ctx = VortexExecutionContext::new();
        match self.transmuter.execute_with_context(&program, &mut ctx) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.enabled = false;
                Err(e)
            }
        }
    }

    pub fn load_code(&mut self, code: Vec<VortexInstruction>) {
        let transmuted = self.transmuter.transmute(&code);
        self.current_code = Some(transmuted);
        self.context = Some(VortexExecutionContext::new());
    }

    pub fn stage_code_swap(&mut self, code: Vec<VortexInstruction>) {
        let transmuted = self.transmuter.transmute(&code);
        self.pending_code_swap = Some(transmuted);
    }

    pub fn try_apply_staged_swap(&mut self) -> bool {
        let Some(staged) = self.pending_code_swap.clone() else {
            return false;
        };

        // Idle actor: swap immediately.
        if self.current_code.is_none() || self.context.is_none() {
            self.current_code = Some(staged);
            self.context = Some(VortexExecutionContext::new());
            self.pending_code_swap = None;
            return true;
        }

        // Mid-execution: only swap at quiescent points (stack depth zero).
        if let (Some(code), Some(ctx)) = (&self.current_code, &self.context) {
            let points = self.transmuter.quiescence_points(code);
            if ctx.done || (ctx.stack_depth == 0 && points.contains(&ctx.pc)) {
                self.current_code = Some(staged);
                self.context = Some(VortexExecutionContext::new());
                self.pending_code_swap = None;
                return true;
            }
        }

        false
    }

    pub fn run(&mut self) -> Result<(), VortexSuspend> {
        if !self.enabled {
            return Err(VortexSuspend);
        }

        if let (Some(code), Some(ctx)) = (&self.current_code, &mut self.context) {
            let result = self.transmuter.execute_with_context(code, ctx);
            if result.is_ok() {
                // When a staged swap exists, apply it on completion; otherwise clear actor state.
                if self.pending_code_swap.is_some() {
                    let _ = self.try_apply_staged_swap();
                } else {
                    self.current_code = None;
                    self.context = None;
                }
            }
            result
        } else {
            Ok(())
        }
    }

    pub fn replenish_budget(&mut self, amount: usize) {
        self.transmuter.instruction_budget += amount;
        self.budget = self.transmuter.instruction_budget;
        self.enabled = true;
    }

    pub fn start_transaction(&mut self, id: u64) {
        self.transaction = Some(VortexTransaction::new(id));
    }

    pub fn start_transaction_with_checkpoint(&mut self, id: u64, locals: HashMap<String, Vec<u8>>) {
        let mut trx = VortexTransaction::new(id);
        trx.checkpoint_locals(locals);
        self.transaction = Some(trx);
    }

    pub fn stage_transaction_vio(&mut self, op: String, payload: Vec<u8>) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.stage_vio(op, payload),
            None => false,
        }
    }

    pub fn start_ghost_transaction_with_checkpoint(
        &mut self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) {
        let mut trx = VortexTransaction::new(id);
        trx.checkpoint_locals(locals);
        self.ghost_transactions.insert(id, trx);
    }

    pub fn stage_ghost_transaction_vio(
        &mut self,
        ghost_id: u64,
        op: String,
        payload: Vec<u8>,
    ) -> bool {
        match self.ghost_transactions.get_mut(&ghost_id) {
            Some(trx) => trx.stage_vio(op, payload),
            None => false,
        }
    }

    pub fn resolve_primary_ghost_race(
        &mut self,
        ghost_id: u64,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let primary = self.transaction.as_mut()?;
        let ghost = self.ghost_transactions.get_mut(&ghost_id)?;

        let result = VortexTransaction::resolve_ghost_race(primary, ghost, winner_id, policy)?;
        self.ghost_transactions.remove(&ghost_id);
        Some(result)
    }

    pub fn replay_committed_vio_calls<F>(&self, calls: &[VortexVioCall], mut executor: F) -> usize
    where
        F: FnMut(&VortexVioCall) -> bool,
    {
        let mut applied = 0usize;
        for call in calls {
            if !executor(call) {
                break;
            }
            applied += 1;
        }
        applied
    }

    pub fn transaction_staged_vio_len(&self) -> usize {
        match &self.transaction {
            Some(trx) => trx.staged_vio_len(),
            None => 0,
        }
    }

    pub fn transaction_committed_vio_len(&self) -> usize {
        match &self.transaction {
            Some(trx) => trx.committed_vio_len(),
            None => 0,
        }
    }

    pub fn take_committed_transaction_vio(&mut self) -> Vec<VortexVioCall> {
        match &mut self.transaction {
            Some(trx) => trx.drain_committed_vio(),
            None => Vec::new(),
        }
    }

    pub fn commit_transaction(&mut self) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.commit(),
            None => false,
        }
    }

    pub fn abort_transaction(&mut self) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.abort(),
            None => false,
        }
    }

    pub fn detach_stalled_thread(&mut self) {
        self.rescue_pool.detach_thread();
    }

    pub fn reclaim_thread(&mut self) {
        self.rescue_pool.reclaim_thread();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // This suspends before BinaryOp and leaves stack_depth > 0.
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

        // First run suspends.
        assert_eq!(engine.run(), Err(VortexSuspend));
        engine.replenish_budget(10);

        // Completion should apply the staged swap and keep current_code alive.
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

        // PreferPrimary must choose primary even if winner_id points to ghost.
        let resolution = engine
            .resolve_primary_ghost_race(4000, 4000, VortexGhostPolicy::PreferPrimary)
            .expect("resolution expected");
        assert_eq!(resolution.winner_id, 3000);
        assert_eq!(resolution.loser_id, 4000);
        assert_eq!(resolution.committed_vio.len(), 1);
        assert_eq!(resolution.committed_vio[0].op, "io_primary");

        // Ghost entry is removed after race resolution.
        let second =
            engine.resolve_primary_ghost_race(4000, 4000, VortexGhostPolicy::PreferPrimary);
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
            seen == 1
        });
        assert_eq!(applied, 1);
    }
}
