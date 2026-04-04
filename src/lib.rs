// src/lib.rs
//! Iris — core runtime (Phase 1-7)
//!
//! This crate contains the core logic for PID allocation, mailboxes,
//! cooperative scheduling, distributed networking, name registration,
//! and remote service discovery.

pub mod buffer;
pub mod mailbox;
pub mod network;
pub mod pid;
pub mod registry;
pub mod supervisor;

#[cfg(feature = "vortex")]
pub mod vortex;

#[cfg(feature = "pyo3")]
pub mod py;

#[cfg(feature = "node")]
pub mod node;

use crate::pid::Pid;
#[cfg(feature = "vortex")]
use crate::vortex::{
    VortexEngine, VortexGhostPolicy, VortexGhostResolution, VortexVioCall,
};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::time::Duration;

type BoxFutureUnit = Pin<Box<dyn Future<Output = ()> + Send>>;
type ErasedMessageHandler = Arc<dyn Fn(mailbox::Message) -> BoxFutureUnit + Send + Sync>;
const MAX_BEHAVIOR_HISTORY: usize = 16;

#[cfg(feature = "vortex")]
fn next_dynamic_budget(
    current: usize,
    base: usize,
    saw_suspend: bool,
    suspend_rate: f64,
    low_thresh: f64,
    high_thresh: f64,
) -> usize {
    let base = base.max(1);
    let min_budget = (base / 4).max(1);
    let max_budget = base.saturating_mul(4).max(base);

    let adjusted = if suspend_rate > high_thresh {
        (current * 60 / 100).max(min_budget)
    } else if suspend_rate > low_thresh {
        (current * 80 / 100).max(min_budget)
    } else if saw_suspend {
        (current / 2).max(min_budget)
    } else {
        (current.saturating_add(1)).min(max_budget)
    };

    adjusted.clamp(min_budget, max_budget)
}

#[derive(Clone)]
struct VirtualActorSpec {
    handler: ErasedMessageHandler,
    budget: usize,
    idle_timeout: Option<Duration>,
}

/// A global, multi-threaded Tokio runtime shared by all Iris instances.
static RUNTIME: Lazy<TokioRuntime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Iris Tokio Runtime")
});

/// Lightweight runtime for spawning actors and managing distributed nodes.
#[derive(Clone)]
pub struct Runtime {
    slab: Arc<Mutex<pid::SlabAllocator>>,
    mailboxes: Arc<DashMap<Pid, mailbox::MailboxSender>>,
    supervisor: Arc<supervisor::Supervisor>,
    observers: Arc<DashMap<Pid, Arc<Mutex<Vec<mailbox::Message>>>>>,
    network: Arc<Mutex<Option<network::NetworkManager>>>,
    // network configuration (timeouts/limits/backoff)
    network_io_timeout: Arc<Mutex<Duration>>,
    network_max_payload: Arc<Mutex<usize>>,
    network_max_name_len: Arc<Mutex<usize>>,
    monitor_backoff_factor: Arc<Mutex<f64>>,
    monitor_backoff_max: Arc<Mutex<Duration>>,
    monitor_failure_threshold: Arc<Mutex<usize>>,
    #[cfg(feature = "vortex")]
    vortex_engine: Option<Arc<Mutex<VortexEngine>>>,
    #[cfg(feature = "vortex")]
    vortex_watcher: Option<Arc<vortex::VortexWatcher>>,
    registry: Arc<registry::NameRegistry>,
    /// Mapping for locally‑spawned proxies that forward to remote actors.
    /// Key is the local proxy PID; value is (remote address, remote PID).
    remote_proxies: Arc<DashMap<Pid, (String, Pid)>>,
    /// Reverse lookup from (address, remote_pid) -> local proxy PID.
    proxy_by_remote: Arc<DashMap<(String, Pid), Pid>>,
    /// Behavior version per actor PID (starts at 1).
    behavior_versions: Arc<DashMap<Pid, u64>>,
    /// Recent hot-swapped pointers used for rollback (capped).
    behavior_history: Arc<DashMap<Pid, Vec<usize>>>,
    /// Optional per-path supervisors (shallow supervisors keyed by path).
    path_supervisors: Arc<DashMap<String, Arc<supervisor::Supervisor>>>,
    /// Maps a child PID to its parent PID for structured concurrency.
    parent_of: Arc<DashMap<Pid, Pid>>,
    /// Tracks direct children of each parent PID.
    children_by_parent: Arc<DashMap<Pid, Vec<Pid>>>,
    /// Capacity for bounded mailboxes.
    bounded_capacity: Arc<DashMap<Pid, usize>>,
    /// Overflow policies for bounded mailboxes; default is DropNew if absent.
    overflow_policy: Arc<DashMap<Pid, mailbox::OverflowPolicy>>,
    /// Lazy/virtual actor specs reserved by PID and activated on first send.
    virtual_specs: Arc<DashMap<Pid, VirtualActorSpec>>,
    /// Per-virtual-actor activation lock to prevent duplicate activation races.
    virtual_activate_locks: Arc<DashMap<Pid, Arc<Mutex<()>>>>,
    /// Track last known backpressure level for each pid, to emit signals on change.
    backpressure_state: Arc<DashMap<Pid, mailbox::BackpressureLevel>>,
    // Runtime-configurable limits for Python GIL-release behavior
    release_gil_max_threads: Arc<Mutex<usize>>,
    gil_pool_size: Arc<Mutex<usize>>,
    release_gil_strict: Arc<Mutex<bool>>,
    // Timers: map from timer id -> cancellation sender
    timers: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<()>>>>,
    timer_counter: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    vortex_ghost_counter: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    vortex_auto_replay_count: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    vortex_auto_primary_wins: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    vortex_auto_ghost_wins: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    vortex_auto_policy: Arc<Mutex<VortexGhostPolicy>>,
    #[cfg(feature = "vortex")]
    vortex_genetic_budgeting_enabled: Arc<Mutex<bool>>,
    #[cfg(feature = "vortex")]
    vortex_genetic_thresholds: Arc<Mutex<(f64, f64)>>,
    #[cfg(feature = "vortex")]
    vortex_isolation_disallowed_ops: Arc<Mutex<std::collections::HashSet<u8>>>,
    #[cfg(feature = "vortex")]
    vortex_genetic_history: Arc<DashMap<Pid, (usize, usize)>>,
}

impl Runtime {
    /// Create a new runtime instance and initialize the networking and registry sub-systems.
    pub fn new() -> Self {
        #[cfg(feature = "pyo3")]
        {
            pyo3::prepare_freethreaded_python();
        }

        let rt = Runtime {
            slab: Arc::new(Mutex::new(pid::SlabAllocator::new())),
            mailboxes: Arc::new(DashMap::new()),
            supervisor: Arc::new(supervisor::Supervisor::new()),
            observers: Arc::new(DashMap::new()),
            network: Arc::new(Mutex::new(None)),
            registry: Arc::new(registry::NameRegistry::new()),
            path_supervisors: Arc::new(DashMap::new()),
            parent_of: Arc::new(DashMap::new()),
            children_by_parent: Arc::new(DashMap::new()),
            bounded_capacity: Arc::new(DashMap::new()),
            overflow_policy: Arc::new(DashMap::new()),
            backpressure_state: Arc::new(DashMap::new()),
            virtual_specs: Arc::new(DashMap::new()),
            virtual_activate_locks: Arc::new(DashMap::new()),
            release_gil_max_threads: Arc::new(Mutex::new(0)),
            gil_pool_size: Arc::new(Mutex::new(8)),
            release_gil_strict: Arc::new(Mutex::new(false)),
            timers: Arc::new(Mutex::new(HashMap::new())),
            timer_counter: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_ghost_counter: Arc::new(AtomicU64::new(1)),
            #[cfg(feature = "vortex")]
            vortex_auto_replay_count: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_primary_wins: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_ghost_wins: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_policy: Arc::new(Mutex::new(VortexGhostPolicy::FirstSafePointWins)),
            #[cfg(feature = "vortex")]
            vortex_genetic_budgeting_enabled: Arc::new(Mutex::new(false)),
            #[cfg(feature = "vortex")]
            vortex_genetic_thresholds: Arc::new(Mutex::new((0.4, 0.7))),
            #[cfg(feature = "vortex")]
            vortex_isolation_disallowed_ops: Arc::new(Mutex::new(std::collections::HashSet::new())),
            #[cfg(feature = "vortex")]
            vortex_genetic_history: Arc::new(DashMap::new()),
            network_io_timeout: Arc::new(Mutex::new(Duration::from_secs(5))),
            network_max_payload: Arc::new(Mutex::new(1024 * 1024)),
            network_max_name_len: Arc::new(Mutex::new(1024)),
            monitor_backoff_factor: Arc::new(Mutex::new(2.0)),
            monitor_backoff_max: Arc::new(Mutex::new(Duration::from_secs(60))),
            monitor_failure_threshold: Arc::new(Mutex::new(1)),
            remote_proxies: Arc::new(DashMap::new()),
            proxy_by_remote: Arc::new(DashMap::new()),
            behavior_versions: Arc::new(DashMap::new()),
            behavior_history: Arc::new(DashMap::new()),
            #[cfg(feature = "vortex")]
            vortex_engine: Some(Arc::new(Mutex::new(VortexEngine::new()))),
            #[cfg(feature = "vortex")]
            vortex_watcher: Some(Arc::new(vortex::VortexWatcher::new())),
        };

        let net_manager = network::NetworkManager::new(Arc::new(rt.clone()));
        *rt.network.lock().unwrap() = Some(net_manager);

        rt
    }

    /// Schedule a one-shot message to be sent after `delay_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the pending send.
    pub fn send_after(&self, pid: Pid, delay_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        RUNTIME.spawn(async move {
            let sleep = tokio::time::sleep(std::time::Duration::from_millis(delay_ms));
            tokio::select! {
                _ = sleep => {
                    let _ = rt_clone.send(pid, msg);
                }
                _ = rx => {
                    // cancelled
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Schedule a repeating interval that sends `msg` every `interval_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the interval.
    pub fn send_interval(&self, pid: Pid, interval_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        RUNTIME.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let _ = rt_clone.send(pid, msg.clone());
                    }
                    _ = &mut rx => {
                        break;
                    }
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Cancel a scheduled timer/interval. Returns true if a timer was cancelled.
    pub fn cancel_timer(&self, timer_id: u64) -> bool {
        if let Some(tx) = self.timers.lock().unwrap().remove(&timer_id) {
            let _ = tx.send(());
            true
        } else {
            false
        }
    }

    /// Set runtime limits for GIL release handling.
    ///
    /// `max_threads = 0` forces pooled mode (no dedicated thread per actor).
    pub fn set_release_gil_limits(&self, max_threads: usize, pool_size: usize) {
        *self.release_gil_max_threads.lock().unwrap() = max_threads;
        *self.gil_pool_size.lock().unwrap() = pool_size;
    }

    /// Enable or disable strict failure mode: when true, spawning an actor with
    /// `release_gil=true` will return an error if the dedicated-thread limit is exceeded.
    pub fn set_release_gil_strict(&self, strict: bool) {
        *self.release_gil_strict.lock().unwrap() = strict;
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_engine(&self) -> Option<VortexEngine> {
        self.vortex_engine
            .as_ref()
            .and_then(|engine| engine.lock().ok().map(|guard| guard.clone()))
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_start_transaction_with_checkpoint(
        &self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => {
                guard.start_transaction_with_checkpoint(id, locals);
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_stage_transaction_vio(&self, op: String, payload: Vec<u8>) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.stage_transaction_vio(op, payload),
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_commit_transaction(&self) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.commit_transaction(),
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_take_committed_transaction_vio(&self) -> Option<Vec<VortexVioCall>> {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return None;
        };
        match engine.lock() {
            Ok(mut guard) => Some(guard.take_committed_transaction_vio()),
            Err(_) => None,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_start_ghost_transaction_with_checkpoint(
        &self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => {
                guard.start_ghost_transaction_with_checkpoint(id, locals);
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_stage_ghost_transaction_vio(
        &self,
        ghost_id: u64,
        op: String,
        payload: Vec<u8>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.stage_ghost_transaction_vio(ghost_id, op, payload),
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_resolve_primary_ghost_race(
        &self,
        ghost_id: u64,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return None;
        };
        match engine.lock() {
            Ok(mut guard) => guard.resolve_primary_ghost_race(ghost_id, winner_id, policy),
            Err(_) => None,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_replay_committed_vio_calls<F>(&self, calls: &[VortexVioCall], executor: F) -> Option<usize>
    where
        F: FnMut(&VortexVioCall) -> bool,
    {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return None;
        };
        match engine.lock() {
            Ok(guard) => Some(guard.replay_committed_vio_calls(calls, executor)),
            Err(_) => None,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_auto_replay_count(&self) -> u64 {
        self.vortex_auto_replay_count.load(Ordering::Relaxed)
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_set_auto_ghost_policy(&self, policy: VortexGhostPolicy) -> bool {
        match self.vortex_auto_policy.lock() {
            Ok(mut guard) => {
                *guard = policy;
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_auto_ghost_policy(&self) -> Option<VortexGhostPolicy> {
        self.vortex_auto_policy.lock().ok().map(|guard| *guard)
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_auto_resolution_counts(&self) -> (u64, u64) {
        (
            self.vortex_auto_primary_wins.load(Ordering::Relaxed),
            self.vortex_auto_ghost_wins.load(Ordering::Relaxed),
        )
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_reset_auto_telemetry(&self) {
        self.vortex_auto_replay_count.store(0, Ordering::Relaxed);
        self.vortex_auto_primary_wins.store(0, Ordering::Relaxed);
        self.vortex_auto_ghost_wins.store(0, Ordering::Relaxed);
        self.vortex_ghost_counter.store(1, Ordering::Relaxed);
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_set_genetic_budgeting(&self, enabled: bool) -> bool {
        match self.vortex_genetic_budgeting_enabled.lock() {
            Ok(mut guard) => {
                *guard = enabled;
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_genetic_budgeting_enabled(&self) -> Option<bool> {
        self.vortex_genetic_budgeting_enabled
            .lock()
            .ok()
            .map(|guard| *guard)
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_set_genetic_thresholds(&self, low: f64, high: f64) -> bool {
        if low < 0.0 || high < 0.0 || low >= high || high > 1.0 {
            return false;
        }
        match self.vortex_genetic_thresholds.lock() {
            Ok(mut guard) => {
                *guard = (low, high);
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_genetic_thresholds(&self) -> Option<(f64, f64)> {
        self.vortex_genetic_thresholds
            .lock()
            .ok()
            .map(|guard| *guard)
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_set_isolation_disallowed_ops(&self, ops: Vec<u8>) -> bool {
        match self.vortex_isolation_disallowed_ops.lock() {
            Ok(mut guard) => {
                guard.clear();
                for op in ops {
                    guard.insert(op);
                }
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_get_isolation_disallowed_ops(&self) -> Option<Vec<u8>> {
        self.vortex_isolation_disallowed_ops
            .lock()
            .ok()
            .map(|guard| guard.iter().copied().collect())
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_watchdog_enable(&self) -> bool {
        if let Some(watcher) = self.vortex_watcher.as_ref() {
            watcher.enable();
            true
        } else {
            false
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_watchdog_disable(&self) -> bool {
        if let Some(watcher) = self.vortex_watcher.as_ref() {
            watcher.disable();
            true
        } else {
            false
        }
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_watchdog_enabled(&self) -> Option<bool> {
        self.vortex_watcher
            .as_ref()
            .map(|w| w.is_enabled())
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_genetic_history(&self, pid: Pid) -> Option<(usize, usize)> {
        self.vortex_genetic_history.get(&pid).map(|r| *r)
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_get_all_genetic_history(&self) -> Vec<(Pid, usize, usize)> {
        self.vortex_genetic_history
            .iter()
            .map(|entry| (*entry.key(), entry.value().0, entry.value().1))
            .collect()
    }

    #[cfg(feature = "vortex")]
    pub fn vortex_reset_genetic_history(&self) {
        self.vortex_genetic_history.clear();
    }

    #[cfg(feature = "vortex")]
    fn vortex_auto_checkpoint_and_replay_on_suspend(&self, pid: Pid, budget: usize) {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return;
        };

        let primary_id = self.vortex_ghost_counter.fetch_add(1, Ordering::Relaxed);
        let ghost_id = self.vortex_ghost_counter.fetch_add(1, Ordering::Relaxed);

        let mut primary_locals = HashMap::new();
        primary_locals.insert("pid".to_string(), pid.to_le_bytes().to_vec());
        primary_locals.insert("budget".to_string(), (budget as u64).to_le_bytes().to_vec());

        let mut ghost_locals = HashMap::new();
        ghost_locals.insert("pid".to_string(), pid.to_le_bytes().to_vec());
        ghost_locals.insert("budget".to_string(), (budget as u64).to_le_bytes().to_vec());

        let Ok(mut guard) = engine.lock() else {
            return;
        };

        guard.start_transaction_with_checkpoint(primary_id, primary_locals);
        let _ = guard.stage_transaction_vio("suspend_primary".to_string(), pid.to_le_bytes().to_vec());

        guard.start_ghost_transaction_with_checkpoint(ghost_id, ghost_locals);
        let _ = guard.stage_ghost_transaction_vio(ghost_id, "suspend_ghost".to_string(), pid.to_le_bytes().to_vec());

        let policy = self
            .vortex_auto_policy
            .lock()
            .map(|guard| *guard)
            .unwrap_or(VortexGhostPolicy::FirstSafePointWins);

        if let Some(resolution) = guard.resolve_primary_ghost_race(ghost_id, ghost_id, policy) {
            if resolution.winner_id == primary_id {
                self.vortex_auto_primary_wins.fetch_add(1, Ordering::Relaxed);
            } else if resolution.winner_id == ghost_id {
                self.vortex_auto_ghost_wins.fetch_add(1, Ordering::Relaxed);
            }

            let applied = guard.replay_committed_vio_calls(&resolution.committed_vio, |_| true);
            if applied > 0 {
                self.vortex_auto_replay_count
                    .fetch_add(applied as u64, Ordering::Relaxed);
            }
        }
    }

    /// Get the current release_gil limits (max_threads, pool_size).
    pub fn get_release_gil_limits(&self) -> (usize, usize) {
        (
            *self.release_gil_max_threads.lock().unwrap(),
            *self.gil_pool_size.lock().unwrap(),
        )
    }

    /// Returns whether strict failure mode is enabled.
    pub fn is_release_gil_strict(&self) -> bool {
        *self.release_gil_strict.lock().unwrap()
    }

    // --- Name Registry ---

    /// Register a name for an actor locally.
    pub fn register(&self, name: String, pid: Pid) {
        self.registry.register(name, pid);
    }

    /// Unregister a named actor locally.
    /// This was missing and caused the compilation error.
    pub fn unregister(&self, name: &str) {
        self.registry.unregister(name);
    }

    /// Resolve a human-readable name to a PID.
    pub fn resolve(&self, name: &str) -> Option<Pid> {
        self.registry.resolve(name)
    }

    /// Set the network I/O timeout used by all operations.
    pub fn set_network_io_timeout(&self, t: Duration) {
        *self.network_io_timeout.lock().unwrap() = t;
    }

    /// Get configured network I/O timeout.
    pub fn get_network_io_timeout(&self) -> Duration {
        *self.network_io_timeout.lock().unwrap()
    }

    /// Adjust maximum allowed payload length for send_remote (bytes).
    pub fn set_network_max_payload(&self, bytes: usize) {
        *self.network_max_payload.lock().unwrap() = bytes;
    }

    /// Get current payload limit.
    pub fn get_network_max_payload(&self) -> usize {
        *self.network_max_payload.lock().unwrap()
    }

    /// Adjust maximum allowed name length for remote resolve.
    pub fn set_network_max_name_len(&self, bytes: usize) {
        *self.network_max_name_len.lock().unwrap() = bytes;
    }

    /// Get current name length limit.
    pub fn get_network_max_name_len(&self) -> usize {
        *self.network_max_name_len.lock().unwrap()
    }

    /// Configure exponential backoff parameters for `monitor_remote`.
    ///
    /// `factor` is multiplied after each failure, capped by `max`.
    /// `failure_threshold` is how many consecutive failures must occur before
    /// the supervisor is notified.
    pub fn set_monitor_backoff(&self, factor: f64, max: Duration, failure_threshold: usize) {
        *self.monitor_backoff_factor.lock().unwrap() = factor;
        *self.monitor_backoff_max.lock().unwrap() = max;
        *self.monitor_failure_threshold.lock().unwrap() = failure_threshold;
    }

    pub fn get_monitor_backoff_factor(&self) -> f64 {
        *self.monitor_backoff_factor.lock().unwrap()
    }
    pub fn get_monitor_backoff_max(&self) -> Duration {
        *self.monitor_backoff_max.lock().unwrap()
    }
    pub fn get_monitor_failure_threshold(&self) -> usize {
        *self.monitor_failure_threshold.lock().unwrap()
    }

    /// Register a hierarchical path for an actor PID.
    pub fn register_path(&self, path: String, pid: Pid) {
        self.registry.register(path, pid);
    }

    /// Unregister a hierarchical path.
    pub fn unregister_path(&self, path: &str) {
        self.registry.unregister(path);
    }

    /// Resolve a path to a PID (exact match).
    pub fn whereis_path(&self, path: &str) -> Option<Pid> {
        self.registry.resolve(path)
    }

    /// Create a path-scoped supervisor for `path`.
    pub fn create_path_supervisor(&self, path: &str) {
        self.path_supervisors
            .entry(path.to_string())
            .or_insert_with(|| Arc::new(supervisor::Supervisor::new()));
    }

    /// Remove a path-scoped supervisor if present.
    pub fn remove_path_supervisor(&self, path: &str) {
        self.path_supervisors.remove(path);
    }

    /// Watch a specific pid under a path-scoped supervisor if it exists,
    /// otherwise fall back to the global supervisor.
    pub fn path_supervisor_watch(&self, path: &str, pid: Pid) {
        if let Some(entry) = self.path_supervisors.get(path) {
            entry.watch(pid);
        } else {
            self.supervisor().watch(pid);
        }
    }

    /// Return child PIDs supervised by the path-scoped supervisor, if any.
    pub fn path_supervisor_children(&self, path: &str) -> Vec<Pid> {
        if let Some(entry) = self.path_supervisors.get(path) {
            entry.child_pids()
        } else {
            Vec::new()
        }
    }

    /// List registered entries under a path prefix.
    pub fn list_children(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.list_children(prefix)
    }

    /// List only direct children one level below `prefix`.
    pub fn list_children_direct(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.list_direct_children(prefix)
    }

    /// Watch all direct children under `prefix` (shallow watch).
    /// This is a convenience to register existing PIDs with the supervisor.
    pub fn watch_path(&self, prefix: &str) {
        let children = self.list_children_direct(prefix);
        for (_path, pid) in children {
            self.supervisor.watch(pid);
        }
    }

    /// Spawn an observed handler and register it under `path`.
    pub fn spawn_with_path_observed(&self, budget: usize, path: String) -> Pid {
        let pid = self.spawn_observed_handler(budget);
        self.register_path(path, pid);
        pid
    }

    /// Send a message to an actor by its registered name.
    pub fn send_named(&self, name: &str, msg: mailbox::Message) -> Result<(), String> {
        if let Some(pid) = self.resolve(name) {
            self.send(pid, msg).map_err(|_| "Send failed".to_string())
        } else {
            Err(format!("Name '{}' not found", name))
        }
    }

    // --- Distributed Networking ---

    /// Enable the node to receive remote messages on the specified TCP address.
    pub fn listen(&self, addr: String) {
        let rt_handle = Arc::new(self.clone());
        RUNTIME.spawn(async move {
            let manager = network::NetworkManager::new(rt_handle);
            match manager.start_server(&addr).await {
                Ok(actual) => tracing::info!(%actual, "node is now listening for remote messages"),
                Err(e) => eprintln!("[Iris] Network Server Error: {}", e),
            }
        });
    }

    /// Helper used internally when the runtime first learns about a remote
    /// actor.  We spawn a tiny local "proxy" actor whose job is to forward
    /// all user messages over the network to the real PID residing on the
    /// remote node.  Proxies behave exactly like ordinary actors from the
    /// caller's perspective (supervision, backpressure, `is_alive`, etc).
    fn lookup_or_create_proxy(&self, addr: &str, remote_pid: Pid) -> Pid {
        // if the caller accidentally passes a proxy PID that's already been
        // created for this address, just reuse it rather than nesting
        // proxies.
        if let Some(entry) = self.remote_proxies.get(&remote_pid) {
            let existing_addr = &entry.value().0;
            if existing_addr == addr {
                return remote_pid;
            }
        }

        // next, check the reverse index so we can return an existing proxy
        // without spawning a new actor.
        if let Some(entry) = self.proxy_by_remote.get(&(addr.to_string(), remote_pid)) {
            return *entry.value();
        }

        let addr_string = addr.to_string();
        let remote_copy = remote_pid;
        let rt_clone = self.clone();
        // clone early so the closure borrow doesn't consume it permanently
        let addr_string_clone = addr_string.clone();

        // spawn a handler that simply relays user messages
        let proxy_pid = self.spawn_actor(move |mut rx| {
            let rt_inner = rt_clone.clone();
            let addr_inner = addr_string_clone.clone();
            async move {
                while let Some(msg) = rx.recv().await {
                    if let mailbox::Message::User(bytes) = msg {
                        // bypass the public `send_remote` path to avoid
                        // creating a second proxy for the same target.  send
                        // directly through the network manager.
                        let manager = network::NetworkManager::new(Arc::new(rt_inner.clone()));
                        let _ = manager
                            .send_remote(&addr_inner, remote_copy, bytes.clone())
                            .await;
                    }
                }
            }
        });

        // record mapping in both directions
        self.remote_proxies
            .insert(proxy_pid, (addr_string.clone(), remote_pid));
        self.proxy_by_remote
            .insert((addr_string.clone(), remote_pid), proxy_pid);

        // automatically monitor the remote node; if it dies the proxy will
        // be stopped which in turn notifies any watchers of the exit.
        self.monitor_remote(addr_string.clone(), proxy_pid);

        proxy_pid
    }

    /// Resolve a name on a remote node.
    /// This is an async call that queries the remote node's registry.  The
    /// `Pid` returned is not the raw remote pid – it is a local proxy that
    /// forwards messages to the remote actor, allowing the caller to treat the
    /// result exactly like a normal local PID.
    pub async fn resolve_remote_async(&self, addr: String, name: String) -> Option<Pid> {
        let manager = network::NetworkManager::new(Arc::new(self.clone()));
        match manager.resolve_remote(&addr, &name).await {
            Ok(0) => None, // Node returned 0, meaning not found
            Ok(pid) => {
                let proxy = self.lookup_or_create_proxy(&addr, pid);
                Some(proxy)
            }
            Err(e) => {
                eprintln!("[Iris] Remote Resolve Error: {}", e);
                None
            }
        }
    }

    /// Send a binary payload to a PID on a specific remote node.
    pub fn send_remote(&self, addr: String, pid: Pid, data: bytes::Bytes) {
        // convert to a proxy so that callers don't need to manage raw remote
        // PIDs themselves.  This will spawn a proxy actor the first time we
        // talk to a particular remote target.
        let local = self.lookup_or_create_proxy(&addr, pid);
        // route using the normal send path so that overflow policies and
        // virtual activation work identically.
        let _ = self.send_user(local, data);
    }

    /// Remote Monitoring with Heartbeat support.
    /// Periodically probes the remote node (default 1s interval) to detect failures.
    pub fn monitor_remote(&self, addr: String, pid: Pid) {
        // if this pid is already a proxy for the same address we can use it
        // directly, otherwise create/look up a proxy for the raw remote PID.
        let local = if let Some(entry) = self.remote_proxies.get(&pid) {
            let existing_addr = &entry.value().0;
            if existing_addr == &addr {
                pid
            } else {
                self.lookup_or_create_proxy(&addr, pid)
            }
        } else {
            self.lookup_or_create_proxy(&addr, pid)
        };

        let rt_handle = Arc::new(self.clone());
        RUNTIME.spawn(async move {
            let manager = network::NetworkManager::new(rt_handle.clone());
            // Probes node health at a 1000ms interval for silent failure detection
            manager.monitor_remote(addr, local, 1000).await;
        });
    }

    // --- Lifecycle & Core Logic ---

    /// Stop an actor by closing its mailbox.
    pub fn stop(&self, pid: Pid) {
        // if this pid is a proxy, clear both maps so lookups don't return
        // stale entries.  DashMap::remove returns the key and value pair.
        if let Some((_key, (addr, rpid))) = self.remote_proxies.remove(&pid) {
            self.proxy_by_remote.remove(&(addr.clone(), rpid));
        }
        self.behavior_versions.remove(&pid);
        self.behavior_history.remove(&pid);

        self.mailboxes.remove(&pid);
        if self.virtual_specs.remove(&pid).is_some() {
            self.virtual_activate_locks.remove(&pid);
            self.supervisor.notify_exit(pid);
            self.slab.lock().unwrap().deallocate(pid);
            self.handle_exit_internal(pid);
        }
    }

    /// Reserve a virtual/lazy actor PID. The actor is activated on first send.
    ///
    /// `idle_timeout` controls auto-shutdown after inactivity once activated.
    pub fn spawn_virtual_handler_with_budget<H, Fut>(
        &self,
        handler: H,
        budget: usize,
        idle_timeout: Option<Duration>,
    ) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();

        let erased: ErasedMessageHandler =
            Arc::new(move |msg: mailbox::Message| Box::pin(handler(msg)));

        self.virtual_specs.insert(
            pid,
            VirtualActorSpec {
                handler: erased,
                budget,
                idle_timeout,
            },
        );
        self.virtual_activate_locks
            .insert(pid, Arc::new(Mutex::new(())));

        pid
    }

    /// Reserve a virtual/lazy actor with default budget and no idle timeout.
    pub fn spawn_virtual_handler<H, Fut>(&self, handler: H) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.spawn_virtual_handler_with_budget(handler, 100, None)
    }

    fn ensure_virtual_actor_active(&self, pid: Pid) -> bool {
        if self.mailboxes.contains_key(&pid) {
            return true;
        }

        let lock = if let Some(lock_entry) = self.virtual_activate_locks.get(&pid) {
            lock_entry.clone()
        } else {
            return false;
        };

        let _guard = lock.lock().unwrap();

        if self.mailboxes.contains_key(&pid) {
            return true;
        }

        let spec = if let Some((_, spec)) = self.virtual_specs.remove(&pid) {
            spec
        } else {
            return self.mailboxes.contains_key(&pid);
        };
        self.virtual_activate_locks.remove(&pid);

        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let handler = spec.handler.clone();
        let budget = spec.budget;
        let idle_timeout = spec.idle_timeout;

        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            let actor_handle = tokio::spawn(async move {
                let mut processed = 0usize;
                loop {
                    let first_msg = if let Some(idle) = idle_timeout {
                        match tokio::time::timeout(idle, rx.recv()).await {
                            Ok(maybe) => maybe,
                            Err(_) => break,
                        }
                    } else {
                        rx.recv().await
                    };

                    let Some(first_msg) = first_msg else {
                        break;
                    };

                    let h = h_loop.clone();
                    (h)(first_msg).await;
                    processed += 1;

                    while processed < budget {
                        match rx.try_recv() {
                            Some(next_msg) => {
                                let h = h_loop.clone();
                                (h)(next_msg).await;
                                processed += 1;
                            }
                            None => break,
                        }
                    }

                    if processed >= budget {
                        processed = 0;
                        tokio::task::yield_now().await;
                    }
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        true
    }

    /// Block the current thread until the actor with `pid` fully exits.
    pub fn wait(&self, pid: Pid) {
        RUNTIME.block_on(async {
            while self.is_alive(pid) {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });
    }

    /// Send a Hot Swap signal to the actor.
    pub fn hot_swap(&self, pid: Pid, handler_ptr: usize) {
        if let Some(sender) = self.mailboxes.get(&pid) {
            if sender
                .send_system(mailbox::SystemMessage::HotSwap(handler_ptr))
                .is_ok()
            {
                if let Some(mut ver) = self.behavior_versions.get_mut(&pid) {
                    *ver += 1;
                } else {
                    // initial behavior is version 1; first successful swap -> v2
                    self.behavior_versions.insert(pid, 2);
                }

                let mut history = self.behavior_history.entry(pid).or_insert_with(Vec::new);
                history.push(handler_ptr);
                if history.len() > MAX_BEHAVIOR_HISTORY {
                    let overflow = history.len() - MAX_BEHAVIOR_HISTORY;
                    history.drain(0..overflow);
                }
            }
        }
    }

    /// Return current behavior version for an actor.
    pub fn behavior_version(&self, pid: Pid) -> u64 {
        self.behavior_versions
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(1)
    }

    /// Roll back behavior by `steps` previously hot-swapped versions.
    ///
    /// Returns the new behavior version on success.
    pub fn rollback_behavior(&self, pid: Pid, steps: usize) -> Result<u64, String> {
        if steps == 0 {
            return Ok(self.behavior_version(pid));
        }

        let target_ptr = {
            let mut history = self
                .behavior_history
                .get_mut(&pid)
                .ok_or_else(|| "rollback failed: no behavior history".to_string())?;

            if history.len() <= steps {
                return Err("rollback failed: insufficient behavior history".to_string());
            }

            let target_idx = history.len() - 1 - steps;
            let target_ptr = history[target_idx];
            history.truncate(target_idx + 1);
            target_ptr
        };

        let sender = self
            .mailboxes
            .get(&pid)
            .ok_or_else(|| "rollback failed: pid not found".to_string())?;

        sender
            .send_system(mailbox::SystemMessage::HotSwap(target_ptr))
            .map_err(|_| "rollback failed: could not send hot swap".to_string())?;

        let next = self
            .behavior_version(pid)
            .saturating_sub(steps as u64)
            .max(1);
        self.behavior_versions.insert(pid, next);
        Ok(next)
    }

    pub fn spawn_actor<H, Fut>(&self, handler: H) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let res = actor_handle.await;

            // Determine exit reason and metadata
            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            // Notify any path-scoped supervisors that supervise this pid
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            // structured concurrency cleanup
            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    /// Bounded mailbox variant of spawn_actor.
    pub fn spawn_actor_bounded<H, Fut>(&self, handler: H, capacity: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::bounded_channel(capacity);
        self.mailboxes.insert(pid, tx.clone());
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        // track capacity and default policy
        self.bounded_capacity.insert(pid, capacity);
        self.overflow_policy
            .insert(pid, mailbox::OverflowPolicy::DropNew);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let res = actor_handle.await;

            // Determine exit reason and metadata
            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            // Notify any path-scoped supervisors that supervise this pid
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            // structured concurrency cleanup
            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    /// Bounded variant of spawn_actor_with_budget.
    pub fn spawn_actor_with_budget_bounded<H, Fut>(
        &self,
        handler: H,
        _budget: usize,
        capacity: usize,
    ) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::bounded_channel(capacity);
        self.mailboxes.insert(pid, tx.clone());
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        // track capacity and default overflow policy
        self.bounded_capacity.insert(pid, capacity);
        self.overflow_policy
            .insert(pid, mailbox::OverflowPolicy::DropNew);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();
        let fut = handler(rx);

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(fut);
            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn spawn_actor_with_budget<H, Fut>(&self, handler: H, _budget: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();
        let fut = handler(rx);

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(fut);
            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn spawn_handler_with_budget<H, Fut>(&self, handler: H, budget: usize) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let handler = std::sync::Arc::new(handler);
        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            #[cfg(feature = "vortex")]
            let rt_vortex_clone = rt_exit_clone.clone();

            #[cfg(feature = "vortex")]
            let mut vortex_engine = rt_exit_clone
                .vortex_engine()
                .unwrap_or_else(|| crate::vortex::VortexEngine::new());

            let actor_handle = tokio::spawn(async move {
                let mut processed = 0usize;
                #[cfg(feature = "vortex")]
                let mut dynamic_budget = budget.max(1);
                #[cfg(not(feature = "vortex"))]
                let dynamic_budget = budget.max(1);
                while let Some(first_msg) = rx.recv().await {
                    #[cfg(feature = "vortex")]
                    if rt_vortex_clone.vortex_watchdog_enabled().unwrap_or(false) {
                        tokio::task::yield_now().await;
                    }

                    #[cfg(feature = "vortex")]
                    let mut saw_suspend_in_cycle = false;
                    #[cfg(not(feature = "vortex"))]
                    let _saw_suspend_in_cycle = false;

                    #[cfg(feature = "vortex")]
                    let enable_genetic_budgeting = rt_vortex_clone
                        .vortex_genetic_budgeting_enabled()
                        .unwrap_or(false);
                    #[cfg(not(feature = "vortex"))]
                    let _enable_genetic_budgeting = false;

                    #[cfg(feature = "vortex")]
                    {
                        if let Err(_) = vortex_engine.preempt_tick() {
                            saw_suspend_in_cycle = true;
                            rt_vortex_clone.vortex_auto_checkpoint_and_replay_on_suspend(pid, budget);
                            vortex_engine.detach_stalled_thread();
                            vortex_engine.replenish_budget(budget);
                            tokio::task::yield_now().await;
                            vortex_engine.reclaim_thread();
                            if enable_genetic_budgeting {
                                let (low, high) = rt_vortex_clone
                                    .vortex_genetic_thresholds()
                                    .unwrap_or((0.4, 0.7));
                                dynamic_budget = next_dynamic_budget(
                                    dynamic_budget,
                                    budget,
                                    saw_suspend_in_cycle,
                                    0.0,
                                    low,
                                    high,
                                );
                            }
                            continue;
                        }
                    }

                    let h = h_loop.clone();
                    (h)(first_msg).await;
                    processed += 1;

                    while processed < dynamic_budget {
                        match rx.try_recv() {
                            Some(next_msg) => {
                                #[cfg(feature = "vortex")]
                                {
                                    if let Err(_) = vortex_engine.preempt_tick() {
                                        saw_suspend_in_cycle = true;
                                        rt_vortex_clone.vortex_auto_checkpoint_and_replay_on_suspend(pid, budget);
                                        vortex_engine.detach_stalled_thread();
                                        vortex_engine.replenish_budget(budget);
                                        tokio::task::yield_now().await;
                                        vortex_engine.reclaim_thread();
                                        if enable_genetic_budgeting {
                                            let (low, high) = rt_vortex_clone
                                                .vortex_genetic_thresholds()
                                                .unwrap_or((0.4, 0.7));
                                            dynamic_budget = next_dynamic_budget(
                                                dynamic_budget,
                                                budget,
                                                saw_suspend_in_cycle,
                                                0.0,
                                                low,
                                                high,
                                            );
                                        }
                                        break;
                                    }
                                }

                                let h = h_loop.clone();
                                (h)(next_msg).await;
                                processed += 1;
                            }
                            None => break,
                        }
                    }

                    if processed >= dynamic_budget {
                        processed = 0;
                        tokio::task::yield_now().await;
                        #[cfg(feature = "vortex")]
                        {
                            if enable_genetic_budgeting {
                                let (suspend_count, total_count) =
                                    rt_vortex_clone.vortex_genetic_history(pid).unwrap_or((0, 0));
                                let total_count = total_count.saturating_add(1);
                                let suspend_count = suspend_count + (saw_suspend_in_cycle as usize);
                                let suspend_rate = if total_count == 0 {
                                    0.0
                                } else {
                                    (suspend_count as f64) / (total_count as f64)
                                };

                                rt_vortex_clone
                                    .vortex_genetic_history
                                    .insert(pid, (suspend_count, total_count));

                                let (low, high) = rt_vortex_clone
                                    .vortex_genetic_thresholds()
                                    .unwrap_or((0.4, 0.7));

                                dynamic_budget = next_dynamic_budget(
                                    dynamic_budget,
                                    budget,
                                    saw_suspend_in_cycle,
                                    suspend_rate,
                                    low,
                                    high,
                                );
                            }
                        }
                    }
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            // structured concurrency cleanup
            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    /// Spawn a new message-handler actor tied to `parent`.
    pub fn spawn_child_handler_with_budget<H, Fut>(
        &self,
        parent: Pid,
        handler: H,
        budget: usize,
    ) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_handler_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    pub fn spawn_observed_handler(&self, _budget: usize) -> Pid {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        let vec = Arc::new(Mutex::new(Vec::new()));
        self.observers.insert(pid, vec.clone());

        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let v_clone = vec.clone();
            let actor_handle = tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    {
                        let mut guard = v_clone.lock().unwrap();
                        guard.push(msg);
                    }

                    while let Some(next_msg) = rx.try_recv() {
                        let mut guard = v_clone.lock().unwrap();
                        guard.push(next_msg);
                    }

                    tokio::task::yield_now().await;
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo {
                        from: pid,
                        reason: reason.clone(),
                        metadata: meta.clone(),
                    };
                    let _ =
                        sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn get_observed_messages(&self, pid: Pid) -> Option<Vec<mailbox::Message>> {
        self.observers
            .get(&pid)
            .map(|entry| entry.value().lock().unwrap().clone())
    }

    /// Remove and return a single observed message matching the predicate.
    /// Used by FFI helpers to implement selective receive for observed actors.
    pub fn take_observed_message_matching<F>(
        &self,
        pid: Pid,
        mut matcher: F,
    ) -> Option<mailbox::Message>
    where
        F: FnMut(&mailbox::Message) -> bool,
    {
        if let Some(entry) = self.observers.get(&pid) {
            let mut guard = entry.value().lock().unwrap();
            if let Some(pos) = guard.iter().position(|m| matcher(m)) {
                return Some(guard.remove(pos));
            }
        }
        None
    }

    fn emit_backpressure_signal(&self, pid: Pid, level: mailbox::BackpressureLevel) {
        let existing = self
            .backpressure_state
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(mailbox::BackpressureLevel::Normal);

        if existing != level {
            self.backpressure_state.insert(pid, level);
        }
    }

    fn update_backpressure_after_enqueue(
        &self,
        pid: Pid,
        sender: &mailbox::MailboxSender,
    ) -> Option<mailbox::BackpressureLevel> {
        // Unbounded mailboxes are always considered Normal, skip map churn.
        let Some(cap) = self.bounded_capacity.get(&pid).map(|entry| *entry.value()) else {
            return None;
        };

        let prev = self
            .backpressure_state
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(mailbox::BackpressureLevel::Normal);
        let level = sender.backpressure_level_with_hysteresis(Some(cap), prev);
        self.emit_backpressure_signal(pid, level);
        Some(level)
    }

    pub fn mailbox_backpressure(&self, pid: Pid) -> Option<mailbox::BackpressureLevel> {
        self.mailboxes.get(&pid).map(|sender| {
            let cap = self.bounded_capacity.get(&pid).map(|entry| *entry.value());
            sender.backpressure_level(cap)
        })
    }

    pub fn send(&self, pid: Pid, msg: mailbox::Message) -> Result<(), mailbox::Message> {
        let _ = self.ensure_virtual_actor_active(pid);

        // apply overflow policy if bounded
        if let Some(cap) = self.bounded_capacity.get(&pid) {
            let size = self.mailbox_size(pid).unwrap_or(0);
            if size >= *cap {
                if let Some(pol) = self.overflow_policy.get(&pid) {
                    match pol.value() {
                        mailbox::OverflowPolicy::DropNew => return Err(msg),
                        mailbox::OverflowPolicy::DropOld => {
                            // tell receiver to drop oldest user message
                            if let Some(sender) = self.mailboxes.get(&pid) {
                                let _ = sender.send_system(mailbox::SystemMessage::DropOld);
                            }
                            for _ in 0..64 {
                                if self.mailbox_size(pid).unwrap_or(0) < *cap {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                        mailbox::OverflowPolicy::Block => {
                            // busy-wait until space appears
                            while self.mailbox_size(pid).unwrap_or(0) >= *cap {
                                std::thread::yield_now();
                            }
                        }
                        mailbox::OverflowPolicy::Redirect(target) => {
                            // forward to target and consider message handled
                            return self.send(*target, msg);
                        }
                        mailbox::OverflowPolicy::Spill(target) => {
                            // send copy to fallback, then proceed with original below
                            let _ = self.send(*target, msg.clone());
                            while self.mailbox_size(pid).unwrap_or(0) >= *cap {
                                std::thread::yield_now();
                            }
                        }
                    }
                } else {
                    return Err(msg);
                }
            }
        }
        let result = if let Some(sender) = self.mailboxes.get(&pid) {
            let res = sender.send(msg);
            if res.is_ok() {
                self.update_backpressure_after_enqueue(pid, sender.value());
            }
            res
        } else {
            Err(msg)
        };

        result
    }

    /// Send with immediate backpressure feedback for bounded mailboxes.
    ///
    /// This avoids an additional map lookup by computing pressure on the same
    /// enqueue path and returning the resulting level to the caller.
    pub fn send_with_backpressure(
        &self,
        pid: Pid,
        msg: mailbox::Message,
    ) -> Result<mailbox::BackpressureLevel, mailbox::Message> {
        self.send(pid, msg)?;
        Ok(self
            .mailbox_backpressure(pid)
            .unwrap_or(mailbox::BackpressureLevel::Normal))
    }

    /// Send user bytes with a fast path that avoids wrapping in `Message` at callsite.
    /// Internal helper that sends raw bytes to an actor mailbox.
    ///
    /// This exists purely for performance; the public `send` API already
    /// routes through this path automatically.  It is **not** exposed to the
    /// Python bindings and is hidden from generated documentation.
    #[doc(hidden)]
    pub fn send_user(&self, pid: Pid, bytes: bytes::Bytes) -> Result<(), bytes::Bytes> {
        let _ = self.ensure_virtual_actor_active(pid);

        // emulate same overflow-policy logic but with raw bytes
        if let Some(cap) = self.bounded_capacity.get(&pid) {
            let size = self.mailbox_size(pid).unwrap_or(0);
            if size >= *cap {
                if let Some(pol) = self.overflow_policy.get(&pid) {
                    match pol.value() {
                        mailbox::OverflowPolicy::DropNew => return Err(bytes),
                        mailbox::OverflowPolicy::DropOld => {
                            if let Some(sender) = self.mailboxes.get(&pid) {
                                let _ = sender.send_system(mailbox::SystemMessage::DropOld);
                            }
                            for _ in 0..64 {
                                if self.mailbox_size(pid).unwrap_or(0) < *cap {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                        mailbox::OverflowPolicy::Block => {
                            while self.mailbox_size(pid).unwrap_or(0) >= *cap {
                                std::thread::yield_now();
                            }
                        }
                        mailbox::OverflowPolicy::Redirect(target) => {
                            return self.send_user(*target, bytes);
                        }
                        mailbox::OverflowPolicy::Spill(target) => {
                            let _ = self.send_user(*target, bytes.clone());
                            while self.mailbox_size(pid).unwrap_or(0) >= *cap {
                                std::thread::yield_now();
                            }
                        }
                    }
                } else {
                    return Err(bytes);
                }
            }
        }
        let result = if let Some(sender) = self.mailboxes.get(&pid) {
            let res = sender.send_user_bytes(bytes);
            if res.is_ok() {
                self.update_backpressure_after_enqueue(pid, sender.value());
            }
            res
        } else {
            Err(bytes)
        };

        result
    }

    /// Send user bytes with immediate backpressure feedback for bounded mailboxes.
    pub fn send_user_with_backpressure(
        &self,
        pid: Pid,
        bytes: bytes::Bytes,
    ) -> Result<mailbox::BackpressureLevel, bytes::Bytes> {
        self.send_user(pid, bytes)?;
        Ok(self
            .mailbox_backpressure(pid)
            .unwrap_or(mailbox::BackpressureLevel::Normal))
    }

    /// Set overflow policy for an existing bounded mailbox.
    pub fn set_overflow_policy(&self, pid: Pid, policy: mailbox::OverflowPolicy) {
        self.overflow_policy.insert(pid, policy);
    }

    /// Return the number of queued user messages for the actor with `pid`.
    pub fn mailbox_size(&self, pid: Pid) -> Option<usize> {
        self.mailboxes.get(&pid).map(|s| s.len())
    }

    pub fn is_alive(&self, pid: Pid) -> bool {
        // proxies are normal actors so the slab check would cover them, but
        // we also treat any registered proxy as alive even if its mailbox has
        // been removed but the slab entry hasn't been cleaned up yet.
        if self.remote_proxies.contains_key(&pid) {
            return true;
        }
        let slab = self.slab.lock().unwrap();
        slab.is_valid(pid)
    }

    pub fn supervisor(&self) -> Arc<supervisor::Supervisor> {
        self.supervisor.clone()
    }

    pub fn supervise(
        &self,
        pid: Pid,
        factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
        strategy: supervisor::RestartStrategy,
    ) {
        let spec = supervisor::ChildSpec { factory, strategy };
        self.supervisor.add_child(pid, spec);
    }

    /// Spawn an actor whose lifetime is tied to `parent`.
    /// When the parent PID exits (normal or crash) the child will be
    /// automatically stopped as well.  The returned PID behaves just like
    /// one created with `spawn_actor`.
    pub fn spawn_child<H, Fut>(&self, parent: Pid, handler: H) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor(handler);
        // record relationships for structured concurrency
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        // if parent is already dead, immediately stop the child
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    /// Same as `spawn_child` but accepts a budget for cooperative scheduling.
    pub fn spawn_child_with_budget<H, Fut>(&self, parent: Pid, handler: H, budget: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    /// Attach a factory-based child spec to a path-scoped supervisor.
    pub fn path_supervise_with_factory(
        &self,
        path: &str,
        pid: Pid,
        factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
        strategy: supervisor::RestartStrategy,
    ) {
        let spec = supervisor::ChildSpec { factory, strategy };
        let entry = self
            .path_supervisors
            .entry(path.to_string())
            .or_insert_with(|| Arc::new(supervisor::Supervisor::new()));
        entry.add_child(pid, spec);
    }

    pub fn link(&self, a: Pid, b: Pid) {
        self.supervisor.link(a, b);
    }

    /// Internal helper invoked when any actor exits to maintain parent/child
    /// state and to enforce structured concurrency.  This is called from
    /// each spawn_* helper after the actor has torn down its mailbox and been
    /// deallocated.
    fn handle_exit_internal(&self, pid: Pid) {
        // DashMap::remove returns (key, value); clean up if this was a proxy
        if let Some((_key, (addr, rpid))) = self.remote_proxies.remove(&pid) {
            self.proxy_by_remote.remove(&(addr.clone(), rpid));
        }
        self.backpressure_state.remove(&pid);
        self.behavior_versions.remove(&pid);
        self.behavior_history.remove(&pid);
        // remove the pid from its parent's child list (if any)
        if let Some((_, parent)) = self.parent_of.remove(&pid) {
            if let Some(mut entry) = self.children_by_parent.get_mut(&parent) {
                entry.retain(|&p| p != pid);
                if entry.is_empty() {
                    self.children_by_parent.remove(&parent);
                }
            }
        }

        // if this pid itself is a parent, kill its current children
        if let Some((_, children)) = self.children_by_parent.remove(&pid) {
            for child in children {
                // drop reverse mapping and close mailbox to stop actor
                let _ = self.parent_of.remove(&child);
                self.mailboxes.remove(&child);
                self.behavior_versions.remove(&child);
                self.behavior_history.remove(&child);
            }
        }
    }

    pub fn unlink(&self, a: Pid, b: Pid) {
        self.supervisor.unlink(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, timeout, Duration};

    #[tokio::test]
    async fn bounded_spawn_rejects_overflow() {
        let rt = Runtime::new();
        // handler that pulls from the mailbox and forwards to a channel for inspection
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // handler will wait on a one-shot signal before reading from the
        // mailbox.  This guarantees the queue holds the first message when the
        // test attempts the second send.
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        let handler = move |mut mailbox: mailbox::MailboxReceiver| {
            let tx = tx.clone();
            let start_rx = start_rx;
            async move {
                // wait until test gives permission to proceed
                let _ = start_rx.await;
                if let Some(msg) = mailbox.recv().await {
                    let _ = tx.send(msg);
                }
            }
        };

        let pid = rt.spawn_actor_bounded(handler, 1);
        // first send success
        assert!(rt
            .send(pid, mailbox::Message::User(b"x".to_vec().into()))
            .is_ok());
        // second send should be dropped (error returned)
        assert!(rt
            .send(pid, mailbox::Message::User(b"y".to_vec().into()))
            .is_err());

        // now tell the actor it may proceed and consume the queued message
        let _ = start_tx.send(());

        // allow the actor to run and verify it received the first message
        sleep(Duration::from_millis(50)).await;
        let got = rx.try_recv().unwrap();
        assert_eq!(got, mailbox::Message::User(b"x".to_vec().into()));
    }

    #[tokio::test]
    async fn send_user_fast_path_roundtrip_and_missing_pid() {
        let rt = Runtime::new();

        let (tx, mut recv_rx) = tokio::sync::mpsc::unbounded_channel();
        let pid = rt.spawn_handler_with_budget(
            move |msg| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(msg);
                }
            },
            32,
        );

        assert!(rt
            .send_user(pid, bytes::Bytes::from_static(b"hello"))
            .is_ok());

        let got = recv_rx.recv().await.expect("message should be received");
        match got {
            mailbox::Message::User(b) => assert_eq!(b.as_ref(), b"hello"),
            _ => panic!("expected user message"),
        }

        rt.stop(pid);
        tokio::time::sleep(Duration::from_millis(20)).await;

        let payload = bytes::Bytes::from_static(b"payload");
        let err = rt
            .send_user(pid, payload.clone())
            .expect_err("send should fail for stopped pid");
        assert_eq!(err, payload);
    }

    #[tokio::test]
    async fn overflow_policy_spill_forwards_and_keeps_primary_delivery() {
        let rt = Runtime::new();

        let (primary_tx, mut primary_rx) = tokio::sync::mpsc::unbounded_channel();
        let (fallback_tx, mut fallback_rx) = tokio::sync::mpsc::unbounded_channel();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let primary = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let primary_tx = primary_tx.clone();
                let start_rx = start_rx;
                async move {
                    let _ = start_rx.await;
                    for _ in 0..2 {
                        if let Some(msg) = mailbox.recv().await {
                            let _ = primary_tx.send(msg);
                        }
                    }
                }
            },
            1,
        );

        let fallback = rt.spawn_actor(move |mut mailbox: mailbox::MailboxReceiver| {
            let fallback_tx = fallback_tx.clone();
            async move {
                if let Some(msg) = mailbox.recv().await {
                    let _ = fallback_tx.send(msg);
                }
            }
        });

        rt.set_overflow_policy(primary, mailbox::OverflowPolicy::Spill(fallback));

        assert!(rt
            .send(primary, mailbox::Message::User(b"p1".to_vec().into()))
            .is_ok());

        let rt_send = rt.clone();
        let send_task = tokio::task::spawn_blocking(move || {
            rt_send
                .send(primary, mailbox::Message::User(b"p2".to_vec().into()))
                .is_ok()
        });

        let spilled = timeout(Duration::from_secs(1), fallback_rx.recv())
            .await
            .expect("spill should reach fallback promptly")
            .expect("fallback should receive spill copy");
        assert_eq!(spilled, mailbox::Message::User(b"p2".to_vec().into()));

        let _ = start_tx.send(());

        let send_ok = timeout(Duration::from_secs(1), send_task)
            .await
            .expect("spill send should complete")
            .expect("send task should join");
        assert!(send_ok, "spill send should report success");

        let first = timeout(Duration::from_secs(1), primary_rx.recv())
            .await
            .expect("primary first receive")
            .expect("primary first message exists");
        let second = timeout(Duration::from_secs(1), primary_rx.recv())
            .await
            .expect("primary second receive")
            .expect("primary second message exists");

        assert_eq!(first, mailbox::Message::User(b"p1".to_vec().into()));
        assert_eq!(second, mailbox::Message::User(b"p2".to_vec().into()));
    }

    #[tokio::test]
    async fn overflow_policy_block_waits_until_capacity_then_succeeds() {
        let rt = Runtime::new();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let pid = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let tx = tx.clone();
                let start_rx = start_rx;
                async move {
                    let _ = start_rx.await;
                    for _ in 0..2 {
                        if let Some(msg) = mailbox.recv().await {
                            let _ = tx.send(msg);
                        }
                    }
                }
            },
            1,
        );

        rt.set_overflow_policy(pid, mailbox::OverflowPolicy::Block);

        assert!(rt
            .send(pid, mailbox::Message::User(b"b1".to_vec().into()))
            .is_ok());

        let rt_send = rt.clone();
        let mut send_task = tokio::task::spawn_blocking(move || {
            rt_send
                .send(pid, mailbox::Message::User(b"b2".to_vec().into()))
                .is_ok()
        });

        assert!(
            timeout(Duration::from_millis(30), &mut send_task)
                .await
                .is_err(),
            "block policy should wait while mailbox is full"
        );

        let _ = start_tx.send(());

        let send_ok = timeout(Duration::from_secs(1), send_task)
            .await
            .expect("blocked send should complete")
            .expect("send task should join");
        assert!(send_ok, "block policy send should report success");

        let first = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("first message receive")
            .expect("first message exists");
        let second = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("second message receive")
            .expect("second message exists");

        assert_eq!(first, mailbox::Message::User(b"b1".to_vec().into()));
        assert_eq!(second, mailbox::Message::User(b"b2".to_vec().into()));
    }

    #[tokio::test]
    async fn mailbox_backpressure_signals_based_on_capacity() {
        let rt = Runtime::new();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let pid = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let start_rx = start_rx;
                async move {
                    // hold consumption until the test has asserted queue pressure.
                    let _ = start_rx.await;
                    while mailbox.recv().await.is_some() {}
                }
            },
            5,
        );

        let mut sent = 0;
        for i in 1..=10 {
            let payload = format!("msg{}", i);
            if rt
                .send(pid, mailbox::Message::User(payload.into_bytes().into()))
                .is_ok()
            {
                sent += 1;
            } else {
                break;
            }
        }

        assert!(sent >= 4, "expected at least 4 messages to be accepted, got {}", sent);

        let level = rt
            .mailbox_backpressure(pid)
            .expect("backpressure should be available");
        assert!(matches!(level, mailbox::BackpressureLevel::High | mailbox::BackpressureLevel::Critical));

        if sent >= 5 {
            assert_eq!(level, mailbox::BackpressureLevel::Critical);
        } else {
            assert_eq!(level, mailbox::BackpressureLevel::High);
        }

        let _ = start_tx.send(());
    }

    #[tokio::test]
    async fn mailbox_backpressure_send_user_path_updates_level() {
        let rt = Runtime::new();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let pid = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let start_rx = start_rx;
                async move {
                    let _ = start_rx.await;
                    while mailbox.recv().await.is_some() {}
                }
            },
            4,
        );

        for i in 0..3 {
            let payload = bytes::Bytes::from(format!("u{}", i));
            assert!(rt.send_user(pid, payload).is_ok());
        }
        assert_eq!(rt.mailbox_backpressure(pid), Some(mailbox::BackpressureLevel::High));

        assert!(rt.send_user(pid, bytes::Bytes::from_static(b"u3")).is_ok());
        assert_eq!(rt.mailbox_backpressure(pid), Some(mailbox::BackpressureLevel::Critical));

        let _ = start_tx.send(());
    }

    #[tokio::test]
    async fn mailbox_backpressure_recovers_to_normal_after_drain() {
        let rt = Runtime::new();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let pid = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let start_rx = start_rx;
                async move {
                    let _ = start_rx.await;
                    while mailbox.recv().await.is_some() {}
                }
            },
            3,
        );

        assert!(rt
            .send(pid, mailbox::Message::User(b"d1".to_vec().into()))
            .is_ok());
        assert!(rt
            .send(pid, mailbox::Message::User(b"d2".to_vec().into()))
            .is_ok());
        assert!(rt
            .send(pid, mailbox::Message::User(b"d3".to_vec().into()))
            .is_ok());
        assert_eq!(rt.mailbox_backpressure(pid), Some(mailbox::BackpressureLevel::Critical));

        let _ = start_tx.send(());

        timeout(Duration::from_secs(1), async {
            loop {
                if rt.mailbox_size(pid) == Some(0) {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("mailbox should drain");

        assert_eq!(rt.mailbox_backpressure(pid), Some(mailbox::BackpressureLevel::Normal));
    }

    #[tokio::test]
    async fn mailbox_backpressure_unknown_pid_is_none() {
        let rt = Runtime::new();
        assert_eq!(rt.mailbox_backpressure(u64::MAX), None);
    }

    #[tokio::test]
    async fn send_with_backpressure_returns_live_level() {
        let rt = Runtime::new();
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();

        let pid = rt.spawn_actor_bounded(
            move |mut mailbox: mailbox::MailboxReceiver| {
                let start_rx = start_rx;
                async move {
                    let _ = start_rx.await;
                    while mailbox.recv().await.is_some() {}
                }
            },
            5,
        );

        let l1 = rt
            .send_with_backpressure(pid, mailbox::Message::User(b"a".to_vec().into()))
            .expect("first send should succeed");
        assert_eq!(l1, mailbox::BackpressureLevel::Normal);

        let l2 = rt
            .send_with_backpressure(pid, mailbox::Message::User(b"b".to_vec().into()))
            .expect("second send should succeed");
        assert_eq!(l2, mailbox::BackpressureLevel::Normal);

        let l3 = rt
            .send_with_backpressure(pid, mailbox::Message::User(b"c".to_vec().into()))
            .expect("third send should succeed");
        assert_eq!(l3, mailbox::BackpressureLevel::Normal);

        let l4 = rt
            .send_with_backpressure(pid, mailbox::Message::User(b"d".to_vec().into()))
            .expect("fourth send should succeed");
        assert_eq!(l4, mailbox::BackpressureLevel::High);

        let l5 = rt
            .send_with_backpressure(pid, mailbox::Message::User(b"e".to_vec().into()))
            .expect("fifth send should succeed");
        assert_eq!(l5, mailbox::BackpressureLevel::Critical);

        let _ = start_tx.send(());
    }

    #[tokio::test]
    async fn send_user_with_backpressure_unknown_pid_returns_err() {
        let rt = Runtime::new();
        let payload = bytes::Bytes::from_static(b"missing");
        let err = rt
            .send_user_with_backpressure(u64::MAX, payload.clone())
            .expect_err("missing pid should return original payload");
        assert_eq!(err, payload);
    }

    #[tokio::test]
    async fn virtual_actor_activates_on_first_send() {
        let rt = Runtime::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let pid = rt.spawn_virtual_handler_with_budget(
            move |msg| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(msg);
                }
            },
            16,
            None,
        );

        assert!(
            rt.mailbox_size(pid).is_none(),
            "virtual actor should be inactive initially"
        );

        assert!(rt
            .send(pid, mailbox::Message::User(b"lazy".to_vec().into()))
            .is_ok());

        let got = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("virtual handler should receive message")
            .expect("message must exist");
        assert_eq!(got, mailbox::Message::User(b"lazy".to_vec().into()));
        assert!(rt.is_alive(pid));
    }

    #[tokio::test]
    async fn virtual_actor_idle_timeout_deactivates_actor() {
        let rt = Runtime::new();

        let pid = rt.spawn_virtual_handler_with_budget(
            move |_msg| async move {},
            8,
            Some(Duration::from_millis(50)),
        );

        assert!(rt
            .send(pid, mailbox::Message::User(b"ping".to_vec().into()))
            .is_ok());

        timeout(Duration::from_secs(1), async {
            loop {
                if !rt.is_alive(pid) {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("virtual actor should stop after idle timeout");
    }

    #[tokio::test]
    async fn stop_unactivated_virtual_actor_deallocates_pid() {
        let rt = Runtime::new();

        let pid = rt.spawn_virtual_handler_with_budget(move |_msg| async move {}, 8, None);
        assert!(rt.is_alive(pid));

        rt.stop(pid);
        sleep(Duration::from_millis(20)).await;

        assert!(!rt.is_alive(pid));
    }

    #[tokio::test]
    async fn behavior_version_increments_on_hot_swap() {
        let rt = Runtime::new();
        let pid = rt.spawn_observed_handler(8);

        assert_eq!(rt.behavior_version(pid), 1);

        rt.hot_swap(pid, 0xA11CE);
        sleep(Duration::from_millis(20)).await;

        assert_eq!(rt.behavior_version(pid), 2);
    }

    #[tokio::test]
    async fn rollback_behavior_replays_previous_handler_ptr() {
        let rt = Runtime::new();
        let pid = rt.spawn_observed_handler(8);

        rt.hot_swap(pid, 0xBEEF);
        rt.hot_swap(pid, 0xCAFE);
        sleep(Duration::from_millis(20)).await;

        assert_eq!(rt.behavior_version(pid), 3);
        let rolled = rt
            .rollback_behavior(pid, 1)
            .expect("rollback should succeed");
        assert_eq!(rolled, 2);
        assert_eq!(rt.behavior_version(pid), 2);

        let swaps = timeout(Duration::from_secs(1), async {
            loop {
                let msgs = rt
                    .get_observed_messages(pid)
                    .expect("observed actor should still exist");
                let mut swaps = Vec::new();
                for msg in msgs {
                    if let mailbox::Message::System(mailbox::SystemMessage::HotSwap(ptr)) = msg {
                        swaps.push(ptr);
                    }
                }
                if swaps.len() >= 3 {
                    break swaps;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("timed out waiting for hot swap replay messages");

        assert_eq!(swaps, vec![0xBEEF, 0xCAFE, 0xBEEF]);
    }

    #[tokio::test]
    async fn rollback_behavior_requires_enough_history() {
        let rt = Runtime::new();
        let pid = rt.spawn_observed_handler(8);

        rt.hot_swap(pid, 0xBEEF);
        sleep(Duration::from_millis(20)).await;

        let err = rt
            .rollback_behavior(pid, 1)
            .expect_err("rollback should fail");
        assert!(err.contains("history"));
    }

    #[cfg(feature = "vortex")]
    #[test]
    fn runtime_vortex_ghost_lifecycle_wrappers_work() {
        let rt = Runtime::new();

        assert!(rt.vortex_start_transaction_with_checkpoint(10, std::collections::HashMap::new()));
        assert!(rt.vortex_stage_transaction_vio("io_primary".to_string(), b"p".to_vec()));

        assert!(rt.vortex_start_ghost_transaction_with_checkpoint(20, std::collections::HashMap::new()));
        assert!(rt.vortex_stage_ghost_transaction_vio(20, "io_ghost_a".to_string(), b"a".to_vec()));
        assert!(rt.vortex_stage_ghost_transaction_vio(20, "io_ghost_b".to_string(), b"b".to_vec()));

        let resolution = rt
            .vortex_resolve_primary_ghost_race(20, 20, crate::vortex::VortexGhostPolicy::FirstSafePointWins)
            .expect("resolution should exist");
        assert_eq!(resolution.winner_id, 20);
        assert_eq!(resolution.committed_vio.len(), 2);

        let mut seen = Vec::new();
        let applied = rt
            .vortex_replay_committed_vio_calls(&resolution.committed_vio, |call| {
                seen.push(call.op.clone());
                true
            })
            .expect("replay should be available");
        assert_eq!(applied, 2);
        assert_eq!(seen, vec!["io_ghost_a".to_string(), "io_ghost_b".to_string()]);

        let second = rt.vortex_resolve_primary_ghost_race(20, 20, crate::vortex::VortexGhostPolicy::FirstSafePointWins);
        assert!(second.is_none());
    }

    #[cfg(feature = "vortex")]
    #[test]
    fn runtime_vortex_commit_and_take_committed_vio() {
        let rt = Runtime::new();
        assert!(rt.vortex_start_transaction_with_checkpoint(30, std::collections::HashMap::new()));
        assert!(rt.vortex_stage_transaction_vio("io_commit".to_string(), b"x".to_vec()));
        assert!(rt.vortex_commit_transaction());

        let committed = rt
            .vortex_take_committed_transaction_vio()
            .expect("engine should exist");
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].op, "io_commit");
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_ghost_wrappers_during_actor_execution() {
        let rt = Runtime::new();
        let pid = rt.spawn_observed_handler(8);

        assert!(rt.send(pid, mailbox::Message::User(b"before".to_vec().into())).is_ok());

        assert!(rt.vortex_start_transaction_with_checkpoint(101, std::collections::HashMap::new()));
        assert!(rt.vortex_stage_transaction_vio("primary_call".to_string(), b"p".to_vec()));
        assert!(rt.vortex_start_ghost_transaction_with_checkpoint(202, std::collections::HashMap::new()));
        assert!(rt.vortex_stage_ghost_transaction_vio(202, "ghost_call".to_string(), b"g".to_vec()));

        let resolution = rt
            .vortex_resolve_primary_ghost_race(202, 202, crate::vortex::VortexGhostPolicy::FirstSafePointWins)
            .expect("resolution should succeed");
        assert_eq!(resolution.winner_id, 202);
        assert_eq!(resolution.committed_vio.len(), 1);

        let applied = rt
            .vortex_replay_committed_vio_calls(&resolution.committed_vio, |_call| true)
            .expect("replay should be available");
        assert_eq!(applied, 1);

        assert!(rt.send(pid, mailbox::Message::User(b"after".to_vec().into())).is_ok());

        let observed = timeout(Duration::from_secs(1), async {
            loop {
                let msgs = rt
                    .get_observed_messages(pid)
                    .expect("observed actor should exist");

                let user_msgs: Vec<Vec<u8>> = msgs
                    .into_iter()
                    .filter_map(|m| match m {
                        mailbox::Message::User(b) => Some(b.to_vec()),
                        _ => None,
                    })
                    .collect();

                if user_msgs.len() >= 2 {
                    break user_msgs;
                }

                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("timed out waiting for observed messages");

        assert!(observed.iter().any(|m| m.as_slice() == b"before"));
        assert!(observed.iter().any(|m| m.as_slice() == b"after"));
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_auto_ghost_hook_triggers_on_preempt_suspend() {
        let rt = Runtime::new();
        let pid = rt.spawn_handler_with_budget(|_msg| async move {}, 8);

        // Drive enough preemption checks to force suspend-path execution.
        for _ in 0..1400 {
            let _ = rt.send(pid, mailbox::Message::User(b"tick".to_vec().into()));
        }

        timeout(Duration::from_secs(2), async {
            loop {
                if rt.vortex_auto_replay_count() > 0 {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("expected auto replay hook to run on preempt suspend");

        assert!(rt.vortex_auto_replay_count() > 0);
        let (primary_wins, ghost_wins) = rt.vortex_auto_resolution_counts();
        assert!(primary_wins + ghost_wins > 0);
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_auto_policy_prefer_primary_updates_counters() {
        let rt = Runtime::new();
        assert!(rt.vortex_set_auto_ghost_policy(crate::vortex::VortexGhostPolicy::PreferPrimary));
        assert_eq!(
            rt.vortex_auto_ghost_policy(),
            Some(crate::vortex::VortexGhostPolicy::PreferPrimary)
        );

        let pid = rt.spawn_handler_with_budget(|_msg| async move {}, 8);
        for _ in 0..1400 {
            let _ = rt.send(pid, mailbox::Message::User(b"tick".to_vec().into()));
        }

        timeout(Duration::from_secs(2), async {
            loop {
                let (primary_wins, ghost_wins) = rt.vortex_auto_resolution_counts();
                if primary_wins > 0 || ghost_wins > 0 {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("expected auto policy counters to update");

        let (primary_wins, ghost_wins) = rt.vortex_auto_resolution_counts();
        assert!(primary_wins > 0);
        assert_eq!(ghost_wins, 0);
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_auto_telemetry_can_reset() {
        let rt = Runtime::new();
        let pid = rt.spawn_handler_with_budget(|_msg| async move {}, 8);

        for _ in 0..1400 {
            let _ = rt.send(pid, mailbox::Message::User(b"tick".to_vec().into()));
        }

        timeout(Duration::from_secs(2), async {
            loop {
                if rt.vortex_auto_replay_count() > 0 {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("expected auto telemetry to increase");

        let (primary_wins, ghost_wins) = rt.vortex_auto_resolution_counts();
        assert!(rt.vortex_auto_replay_count() > 0);
        assert!(primary_wins + ghost_wins > 0);

        rt.vortex_reset_auto_telemetry();
        let (primary_wins_after, ghost_wins_after) = rt.vortex_auto_resolution_counts();
        assert_eq!(rt.vortex_auto_replay_count(), 0);
        assert_eq!(primary_wins_after, 0);
        assert_eq!(ghost_wins_after, 0);
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_genetic_budgeting_toggle_roundtrip() {
        let rt = Runtime::new();
        assert_eq!(rt.vortex_genetic_budgeting_enabled(), Some(false));
        assert!(rt.vortex_set_genetic_budgeting(true));
        assert_eq!(rt.vortex_genetic_budgeting_enabled(), Some(true));
        assert!(rt.vortex_set_genetic_budgeting(false));
        assert_eq!(rt.vortex_genetic_budgeting_enabled(), Some(false));
    }

    #[cfg(feature = "vortex")]
    #[tokio::test]
    async fn runtime_vortex_genetic_budgeting_effects_during_actor_run() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rt = Runtime::new();

        // Start with dynamic budgeting disabled.
        assert!(rt.vortex_set_genetic_budgeting(false));

        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let pid = rt.spawn_handler_with_budget(
            move |_msg| {
                let counter_clone = counter_clone.clone();
                async move {
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                }
            },
            8,
        );

        for _ in 0..400 {
            let _ = rt.send(pid, mailbox::Message::User(b"x".to_vec().into()));
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Enable genetic budgeting while actor is still running.
        assert!(rt.vortex_set_genetic_budgeting(true));

        for _ in 0..400 {
            let _ = rt.send(pid, mailbox::Message::User(b"x".to_vec().into()));
        }

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if counter.load(Ordering::Relaxed) >= 800 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("all messages should be processed");

        rt.stop(pid);
    }

    #[cfg(feature = "vortex")]
    #[test]
    fn runtime_vortex_genetic_budget_policy_math() {
        // base=8 => min=2, max=32
        assert_eq!(next_dynamic_budget(8, 8, false, 0.0, 0.4, 0.7), 9);
        assert_eq!(next_dynamic_budget(32, 8, false, 0.0, 0.4, 0.7), 32);
        assert_eq!(next_dynamic_budget(8, 8, true, 0.0, 0.4, 0.7), 4);
        assert_eq!(next_dynamic_budget(3, 8, true, 0.0, 0.4, 0.7), 2);
        assert_eq!(next_dynamic_budget(1, 8, true, 0.0, 0.4, 0.7), 2);

        // high suspend rate should force stronger penalty.
        assert_eq!(next_dynamic_budget(16, 8, false, 0.8, 0.4, 0.7), 9);
        assert_eq!(next_dynamic_budget(20, 8, false, 0.5, 0.4, 0.7), 16);
    }

    #[cfg(feature = "vortex")]
    #[test]
    fn runtime_vortex_genetic_threshold_roundtrip() {
        let rt = Runtime::new();
        assert_eq!(rt.vortex_genetic_thresholds(), Some((0.4, 0.7)));
        assert!(rt.vortex_set_genetic_thresholds(0.2, 0.5));
        assert_eq!(rt.vortex_genetic_thresholds(), Some((0.2, 0.5)));
        assert!(!rt.vortex_set_genetic_thresholds(0.7, 0.2));
        assert!(!rt.vortex_set_genetic_thresholds(-0.1, 0.5));
        assert!(!rt.vortex_set_genetic_thresholds(0.1, 1.2));
    }

    #[cfg(feature = "vortex")]
    #[test]
    fn runtime_vortex_isolation_disallow_roundtrip() {
        let rt = Runtime::new();
        assert_eq!(rt.vortex_get_isolation_disallowed_ops(), Some(vec![]));
        assert!(rt.vortex_set_isolation_disallowed_ops(vec![90, 91]));
        let mut got = rt.vortex_get_isolation_disallowed_ops().unwrap();
        got.sort();
        assert_eq!(got, vec![90, 91]);
    }
}
