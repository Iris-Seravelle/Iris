// src/vortex/scheduler.rs
//! Experimental Vortex scheduler placeholder.

pub struct VortexScheduler;

impl VortexScheduler {
    pub fn new() -> Self {
        VortexScheduler
    }

    pub fn describe(&self) -> &'static str {
        "vortex scheduler (stub)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vortex_scheduler_describe() {
        let sched = VortexScheduler::new();
        assert_eq!(sched.describe(), "vortex scheduler (stub)");
    }
}
