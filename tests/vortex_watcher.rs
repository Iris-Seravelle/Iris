#![cfg(feature = "vortex")]

use iris::vortex::VortexWatcher;

#[test]
fn vortex_watcher_health() {
    let watcher = VortexWatcher::new();
    assert_eq!(watcher.health(), "vortex watcher healthy");
}

#[tokio::test]
async fn vortex_watcher_enable_disable() {
    let watcher = VortexWatcher::new();
    assert!(!watcher.is_enabled());
    watcher.enable();
    assert!(watcher.is_enabled());
    watcher.disable();
    assert!(!watcher.is_enabled());
}
