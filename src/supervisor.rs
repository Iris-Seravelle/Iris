// src/supervisor.rs
//! Supervisor
//!
//! Adds small, testable supervision behaviors used by the runtime. Each child
//! can be registered with a `ChildSpec` (factory + restart strategy). When a
//! watched child exits the supervisor may restart the single child (one-for-one)
//! or restart the whole supervised group (one-for-all).

use crate::pid::Pid;
use dashmap::{DashMap, DashSet};
use std::sync::{Arc, Mutex};

/// Restart strategies supported in Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Restart only the failed child.
    RestartOne,
    /// Restart all children supervised by this supervisor.
    RestartAll,
}

/// A child specification holds a factory used to (re)spawn the child and the
/// restart strategy to apply when the child exits.
#[derive(Clone)]
pub struct ChildSpec {
    /// The factory may fail; we return Result<Pid, String> so callers can
    /// surface human-friendly error messages when a factory invocation fails.
    pub factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
    pub strategy: RestartStrategy,
}

/// Supervisor behavior notes:
/// - Factories are fallible and return `Result<Pid,String>`; when a factory
///   fails during a restart we log the failure and skip restarting that child.
/// - This design prevents panics during supervisor restarts caused by Python
///   or other foreign code used as factories; callers should ensure factories
///   return informative error strings to ease debugging.

/// Supervisor stores child specs keyed by `Pid`.
#[derive(Default)]
pub struct Supervisor {
    // Wrapped in Arc so they can be shared with background restart tasks
    children: Arc<DashMap<Pid, ChildSpec>>,
    /// Recent errors recorded while attempting to restart children.
    errors: Arc<Mutex<Vec<String>>>,
    /// Bidirectional links between PIDs. If A is linked to B, and A exits,
    /// B should receive an exit signal (delivered by the Runtime).
    links: Arc<DashMap<Pid, Vec<Pid>>>,
    /// Tracks PIDs currently undergoing a restart to debounce duplicate exit signals
    /// and prevent cascading `RestartAll` loops.
    restarting: Arc<DashSet<Pid>>,
}

impl Supervisor {
    fn push_unique_link(links: &DashMap<Pid, Vec<Pid>>, a: Pid, b: Pid) {
        let mut entry = links.entry(a).or_insert_with(Vec::new);
        if !entry.contains(&b) {
            entry.push(b);
        }
    }

    /// Create a supervisor instance.
    pub fn new() -> Self {
        Supervisor {
            children: Arc::new(DashMap::new()),
            errors: Arc::new(Mutex::new(Vec::new())),
            links: Arc::new(DashMap::new()),
            restarting: Arc::new(DashSet::new()),
        }
    }

    /// Add a child with an explicit child spec (factory + strategy).
    pub fn add_child(&self, pid: Pid, spec: ChildSpec) {
        self.children.insert(pid, spec);
    }

    /// Remove a child from supervision.
    pub fn remove_child(&self, pid: Pid) {
        self.children.remove(&pid);
        self.restarting.remove(&pid);
        self.cleanup_links_internal(pid);
    }

    /// Remove a bidirectional link between two PIDs.
    pub fn unlink(&self, a: Pid, b: Pid) {
        if let Some(mut entry) = self.links.get_mut(&a) {
            entry.retain(|&p| p != b);
            if entry.is_empty() {
                drop(entry);
                self.links.remove(&a);
            }
        }
        if let Some(mut entry) = self.links.get_mut(&b) {
            entry.retain(|&p| p != a);
            if entry.is_empty() {
                drop(entry);
                self.links.remove(&b);
            }
        }
    }

    /// Backwards-compatible `watch` that simply inserts a default ChildSpec.
    /// Useful for tests / simple use-cases.
    pub fn watch(&self, pid: Pid) {
        let spec = ChildSpec {
            factory: Arc::new(move || Ok(pid)),
            strategy: RestartStrategy::RestartOne,
        };
        self.children.insert(pid, spec);
    }

    /// Establish a bidirectional link between two PIDs.
    pub fn link(&self, a: Pid, b: Pid) {
        Self::push_unique_link(&self.links, a, b);
        Self::push_unique_link(&self.links, b, a);
    }

    /// Retrieve and remove the PIDs linked to `pid`.
    ///
    /// This method is destructive: it assumes the actor `pid` is dead or dying.
    /// It removes `pid` from the links map and also removes `pid` from the
    /// link lists of all its peers to prevent memory leaks and stale references.
    pub fn linked_pids(&self, pid: Pid) -> Vec<Pid> {
        if let Some((_, linked_peers)) = self.links.remove(&pid) {
            for peer in &linked_peers {
                if let Some(mut entry) = self.links.get_mut(peer) {
                    entry.retain(|&p| p != pid);
                }
            }
            linked_peers
        } else {
            Vec::new()
        }
    }

    /// Internal helper to cleanup links without returning them.
    fn cleanup_links_internal(&self, pid: Pid) {
        if let Some((_, linked_peers)) = self.links.remove(&pid) {
            for peer in linked_peers {
                if let Some(mut entry) = self.links.get_mut(&peer) {
                    entry.retain(|&p| p != pid);
                }
            }
        }
    }

    /// Stop watching a pid.
    pub fn unwatch(&self, pid: Pid) {
        self.children.remove(&pid);
        self.restarting.remove(&pid);
    }

    /// Query helpers for tests/observability.
    pub fn contains_child(&self, pid: Pid) -> bool {
        self.children.contains_key(&pid)
    }

    pub fn children_count(&self) -> usize {
        self.children.len()
    }

    pub fn child_pids(&self) -> Vec<Pid> {
        self.children.iter().map(|kv| *kv.key()).collect()
    }

    /// Return a snapshot of recent supervisor error messages.
    pub fn errors(&self) -> Vec<String> {
        self.errors.lock().unwrap().clone()
    }

    /// Called by the runtime when a child exits. Applies the restart strategy
    /// recorded in the child's `ChildSpec` (if any).
    pub fn notify_exit(&self, pid: Pid) {
        // Debounce: If we are already restarting this PID, safely ignore the duplicate exit signal.
        if !self.restarting.insert(pid) {
            return;
        }

        let spec = match self.children.get(&pid) {
            Some(s) => s.clone(),
            None => {
                self.restarting.remove(&pid);
                return;
            }
        };

        tracing::info!(
            "[supervisor] notify_exit(pid={}) strategy={:?}",
            pid,
            spec.strategy
        );

        let children = self.children.clone();
        let errors = self.errors.clone();
        let links = self.links.clone();
        let restarting = self.restarting.clone();

        match spec.strategy {
            RestartStrategy::RestartAll => {
                let all: Vec<(Pid, ChildSpec)> = children
                    .iter()
                    .map(|kv| (*kv.key(), kv.value().clone()))
                    .collect();

                // Mark the entire group as restarting to prevent cascaded exit signals
                // from spawning redundant RestartAll waves.
                for (p, _) in &all {
                    restarting.insert(*p);
                }

                // Spawn concurrent restart tasks without dropping the supervisor's registry count.
                for (orig_pid, s) in all {
                    let children_clone = children.clone();
                    let errors_clone = errors.clone();
                    let links_clone = links.clone();
                    let restarting_clone = restarting.clone();

                    tokio::spawn(async move {
                        let mut attempts = 0;
                        let max_attempts = 3;
                        let mut backoff_ms = 100;

                        loop {
                            attempts += 1;
                            match (s.factory)() {
                                Ok(new_pid) => {
                                    // Atomic swap: insert the new PID, then clean up the old one.
                                    children_clone.insert(new_pid, s.clone());
                                    children_clone.remove(&orig_pid);

                                    if let Some((_, v)) = links_clone.remove(&orig_pid) {
                                        for other in v {
                                            if let Some(mut entry) = links_clone.get_mut(&other) {
                                                entry.retain(|&p| p != orig_pid);
                                            }
                                        }
                                    }
                                    restarting_clone.remove(&orig_pid);
                                    break;
                                }
                                Err(err) => {
                                    tracing::error!("[supervisor] factory failed during RestartAll attempt={} err={}", attempts, err);
                                    {
                                        let mut guard = errors_clone.lock().unwrap();
                                        guard.push(err.clone());
                                    }

                                    if attempts >= max_attempts {
                                        tracing::error!("[supervisor] child permanently dropped after exhausting retries (RestartAll) err={}", err);
                                        children_clone.remove(&orig_pid);
                                        if let Some((_, v)) = links_clone.remove(&orig_pid) {
                                            for other in v {
                                                if let Some(mut entry) = links_clone.get_mut(&other)
                                                {
                                                    entry.retain(|&p| p != orig_pid);
                                                }
                                            }
                                        }
                                        restarting_clone.remove(&orig_pid);
                                        break;
                                    }

                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ))
                                    .await;
                                    backoff_ms = backoff_ms.saturating_mul(2);
                                }
                            }
                        }
                    });
                }
            }
            RestartStrategy::RestartOne => {
                let children_clone = children.clone();
                let errors_clone = errors.clone();
                let links_clone = links.clone();
                let restarting_clone = restarting.clone();

                tokio::spawn(async move {
                    let mut attempts = 0;
                    let max_attempts = 3;
                    let mut backoff_ms = 100;

                    loop {
                        attempts += 1;
                        match (spec.factory)() {
                            Ok(new_pid) => {
                                // Atomic swap
                                children_clone.insert(new_pid, spec.clone());
                                children_clone.remove(&pid);

                                if let Some((_, v)) = links_clone.remove(&pid) {
                                    for other in v {
                                        if let Some(mut entry) = links_clone.get_mut(&other) {
                                            entry.retain(|&p| p != pid);
                                        }
                                    }
                                }
                                restarting_clone.remove(&pid);
                                break;
                            }
                            Err(err) => {
                                tracing::error!("[supervisor] factory failed during RestartOne attempt={} err={}", attempts, err);
                                {
                                    let mut guard = errors_clone.lock().unwrap();
                                    guard.push(err.clone());
                                }

                                if attempts >= max_attempts {
                                    tracing::error!("[supervisor] child permanently dropped after exhausting retries (RestartOne) err={}", err);
                                    children_clone.remove(&pid);
                                    if let Some((_, v)) = links_clone.remove(&pid) {
                                        for other in v {
                                            if let Some(mut entry) = links_clone.get_mut(&other) {
                                                entry.retain(|&p| p != pid);
                                            }
                                        }
                                    }
                                    restarting_clone.remove(&pid);
                                    break;
                                }

                                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms))
                                    .await;
                                backoff_ms = backoff_ms.saturating_mul(2);
                            }
                        }
                    }
                });
            }
        }
    }
}
