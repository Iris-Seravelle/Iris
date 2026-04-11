use iris::supervisor::{ChildSpec, RestartStrategy, Supervisor};
use std::sync::Arc;

#[test]
fn watch_and_unwatch() {
    let _ = tracing_subscriber::fmt::try_init();
    let s = Supervisor::new();
    s.watch(1);
    assert!(s.contains_child(1));
    s.unwatch(1);
    assert!(!s.contains_child(1));
}

#[tokio::test]
async fn factory_failure_skips_restart() {
    let _ = tracing_subscriber::fmt::try_init();
    let s = Supervisor::new();
    let bad_factory = Arc::new(move || Err::<u64, String>("boom".to_string()));
    let spec = ChildSpec {
        factory: bad_factory,
        strategy: RestartStrategy::RestartOne,
    };
    s.add_child(42, spec);

    s.notify_exit(42);

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        attempts += 1;

        let no_children = s.children_count() == 0;
        let has_errors = !s.errors().is_empty();

        if no_children && has_errors {
            break;
        }

        assert!(
            attempts <= 30,
            "Timeout waiting for supervisor: children_count={} errors={}",
            s.children_count(),
            s.errors().len()
        );
    }

    let errs = s.errors();
    assert!(errs[0].contains("boom"));
}

#[test]
fn link_is_deduplicated_for_same_pair() {
    let s = Supervisor::new();

    s.link(10, 20);
    s.link(10, 20);
    s.link(20, 10);

    let linked = s.linked_pids(10);
    assert_eq!(linked, vec![20]);

    let reverse = s.linked_pids(20);
    assert!(reverse.is_empty());
}
