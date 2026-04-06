#![cfg(feature = "vortex")]

use iris::vortex::RescuePool;

#[test]
fn rescue_pool_detach_and_reclaim() {
    let mut pool = RescuePool::new();
    assert_eq!(pool.active_count, 0);
    pool.detach_thread();
    assert_eq!(pool.active_count, 1);
    pool.reclaim_thread();
    assert_eq!(pool.active_count, 0);
    pool.reclaim_thread();
    assert_eq!(pool.active_count, 0);
}
