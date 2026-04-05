#[cfg(feature = "vortex")]
use iris::Runtime;
#[cfg(feature = "vortex")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[cfg(feature = "vortex")]
#[tokio::test]
async fn vortex_actor_preemption_and_resume() {
    let rt = Runtime::new();
    let mut engine = rt.vortex_engine().expect("vortex engine should exist");

    assert!(engine.is_enabled());

    engine.set_budget(1);
    engine.load_code(vec![
        iris::vortex::VortexInstruction::LoadFast(0),
        iris::vortex::VortexInstruction::BinaryOp(0),
        iris::vortex::VortexInstruction::ReturnValue,
    ]);

    let first = engine.run();
    assert_eq!(first, Err(iris::vortex::VortexSuspend));

    engine.replenish_budget(5);
    assert_eq!(engine.run(), Ok(()));
}

#[cfg(feature = "vortex")]
#[tokio::test]
async fn vortex_operator_dispatch_preempts_actor_loop() {
    let rt = Runtime::new();
    let counter = Arc::new(AtomicUsize::new(0));
    let c2 = counter.clone();

    let pid = rt.spawn_handler_with_budget(
        move |_msg| {
            let counter2 = c2.clone();
            async move {
                counter2.fetch_add(1, Ordering::SeqCst);
            }
        },
        1,
    );

    assert!(rt.is_alive(pid));

    for _ in 0..5 {
        rt.send(
            pid,
            iris::mailbox::Message::User(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();
    }

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    assert_eq!(counter.load(Ordering::SeqCst), 5);

    // Rescue pool should have returned to 0 after backoff/reclaim cycles.
    let engine = rt.vortex_engine().expect("vortex engine should exist");
    assert_eq!(engine.rescue_pool.active_count, 0);
}

#[cfg(feature = "vortex")]
#[tokio::test]
async fn vortex_infinite_loop_preempts_without_hang() {
    let rt = Runtime::new();
    let mut engine = rt.vortex_engine().expect("vortex engine should exist");

    engine.set_budget(1);
    engine.load_code(vec![iris::vortex::VortexInstruction::JumpBackward(0)]);

    // must not spin forever; should suspend because of injected reduction check.
    assert_eq!(engine.run(), Err(iris::vortex::VortexSuspend));
    engine.replenish_budget(1);
    assert_eq!(engine.run(), Err(iris::vortex::VortexSuspend));
}
