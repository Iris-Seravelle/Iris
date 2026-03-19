// src/vortex/engine.rs
//! Experimental Vortex runtime engine.

use crate::vortex::rescue_pool::RescuePool;
use crate::vortex::transmuter::{VortexExecutionContext, VortexInstruction, VortexSuspend, VortexTransmuter};
use crate::vortex::transaction::VortexTransaction;

#[derive(Debug, Clone)]
pub struct VortexEngine {
    pub enabled: bool,
    pub transmuter: VortexTransmuter,
    pub transaction: Option<VortexTransaction>,
    pub rescue_pool: RescuePool,
    pub budget: usize,
    pub current_code: Option<Vec<VortexInstruction>>,
    pub context: Option<VortexExecutionContext>,
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

    pub fn run(&mut self) -> Result<(), VortexSuspend> {
        if !self.enabled {
            return Err(VortexSuspend);
        }

        if let (Some(code), Some(ctx)) = (&self.current_code, &mut self.context) {
            let result = self.transmuter.execute_with_context(code, ctx);
            if result.is_ok() {
                self.current_code = None;
                self.context = None;
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
}

