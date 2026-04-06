#![cfg(feature = "vortex")]

use iris::vortex::VortexScheduler;

#[test]
fn vortex_scheduler_describe() {
    let sched = VortexScheduler::new();
    assert_eq!(sched.describe(), "vortex scheduler (stub)");
}
