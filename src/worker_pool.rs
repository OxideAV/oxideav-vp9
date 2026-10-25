//! A persistent worker pool for the decoder's per-frame jobs.
//!
//! The pool lives as long as the [`crate::Vp9SequenceDecoder`] that
//! owns it, so no thread is spawned per frame. Jobs are `'static`
//! closures executed in FIFO order; [`WorkerPool::run`] queues a batch,
//! lets the calling thread work the queue alongside the helpers, and
//! returns once every job of the batch has finished.
//!
//! Jobs of one batch may block on progress published by jobs queued
//! *earlier* in the same batch (the frame pipeline's loop-filter rows
//! wait on tile rows and on the row above). FIFO dequeueing makes that
//! deadlock-free at any pool size: when a job is running, every job
//! queued before it has been dequeued, so the earliest unfinished job is
//! always running and only ever waits on finished work. With no helper
//! threads the caller simply runs the batch in order.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Queue {
    jobs: VecDeque<Job>,
    /// Jobs of the current batch not yet finished (queued or running).
    pending: usize,
    /// A job of the current batch panicked; its payload is re-raised on
    /// the caller once the batch drains.
    panic: Option<Box<dyn std::any::Any + Send + 'static>>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when a job is queued (helpers wait on it).
    work: Condvar,
    /// Signalled when the batch's last job finishes.
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run one job, recording a panic instead of unwinding the worker.
    fn execute(&self, job: Job) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
        let mut q = self.lock();
        if let Err(payload) = result
            && q.panic.is_none()
        {
            q.panic = Some(payload);
        }
        q.pending -= 1;
        if q.pending == 0 {
            self.done.notify_all();
        }
    }
}

/// The persistent pool: `helpers` background threads plus the calling
/// thread while it is inside [`WorkerPool::run`].
pub(crate) struct WorkerPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("helpers", &self.threads.len())
            .finish()
    }
}

impl WorkerPool {
    /// A pool with `helpers` background threads (`0` = the caller runs
    /// every job itself).
    pub(crate) fn new(helpers: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                pending: 0,
                panic: None,
                shutdown: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let threads = (0..helpers)
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("vp9-worker-{i}"))
                    .spawn(move || helper_loop(&shared))
                    .expect("spawn vp9 worker thread")
            })
            .collect();
        Self { shared, threads }
    }

    /// Background threads in the pool.
    pub(crate) fn helpers(&self) -> usize {
        self.threads.len()
    }

    /// Run `jobs` to completion (FIFO; the caller participates). A
    /// panicking job's payload is re-raised here after the batch drains.
    pub(crate) fn run(&self, jobs: Vec<Job>) {
        if jobs.is_empty() {
            return;
        }
        {
            let mut q = self.shared.lock();
            debug_assert_eq!(q.pending, 0, "one batch at a time");
            q.pending = jobs.len();
            q.jobs.extend(jobs);
        }
        self.shared.work.notify_all();
        // Work the queue alongside the helpers.
        loop {
            let job = self.shared.lock().jobs.pop_front();
            match job {
                Some(job) => self.shared.execute(job),
                None => break,
            }
        }
        let mut q = self.shared.lock();
        while q.pending > 0 {
            q = self.shared.done.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        if let Some(payload) = q.panic.take() {
            drop(q);
            std::panic::resume_unwind(payload);
        }
    }
}

fn helper_loop(shared: &Shared) {
    loop {
        let job = {
            let mut q = shared.lock();
            loop {
                if let Some(job) = q.jobs.pop_front() {
                    break job;
                }
                if q.shutdown {
                    return;
                }
                q = shared.work.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        shared.execute(job);
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.work.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn runs_every_job_at_every_size() {
        for helpers in [0, 1, 3, 8] {
            let pool = WorkerPool::new(helpers);
            for _ in 0..3 {
                let hits = Arc::new(AtomicUsize::new(0));
                let jobs: Vec<Job> = (0..20)
                    .map(|_| {
                        let hits = Arc::clone(&hits);
                        Box::new(move || {
                            hits.fetch_add(1, Ordering::Relaxed);
                        }) as Job
                    })
                    .collect();
                pool.run(jobs);
                assert_eq!(hits.load(Ordering::Relaxed), 20);
            }
        }
    }

    /// Jobs waiting on earlier jobs of the batch never deadlock, even
    /// with fewer threads than jobs.
    #[test]
    fn chained_waits_complete_with_one_thread() {
        for helpers in [0, 1, 2] {
            let pool = WorkerPool::new(helpers);
            let state = Arc::new((Mutex::new(0usize), Condvar::new()));
            let jobs: Vec<Job> = (0..12)
                .map(|i| {
                    let state = Arc::clone(&state);
                    Box::new(move || {
                        let (m, cv) = &*state;
                        let mut g = m.lock().unwrap();
                        while *g != i {
                            g = cv.wait(g).unwrap();
                        }
                        *g += 1;
                        cv.notify_all();
                    }) as Job
                })
                .collect();
            pool.run(jobs);
            assert_eq!(*state.0.lock().unwrap(), 12);
        }
    }

    #[test]
    fn job_panic_resurfaces_on_the_caller() {
        let pool = WorkerPool::new(2);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let boom: Job = Box::new(|| {
                let fail = true;
                assert!(!fail, "boom");
            });
            pool.run(vec![boom, Box::new(|| {})]);
        }));
        assert!(r.is_err());
        // The pool still works afterwards.
        let hit = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&hit);
        pool.run(vec![Box::new(move || {
            h.fetch_add(1, Ordering::Relaxed);
        })]);
        assert_eq!(hit.load(Ordering::Relaxed), 1);
    }
}
