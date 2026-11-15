//! A persistent worker pool for the decoder's per-frame jobs.
//!
//! The pool lives as long as the [`crate::Vp9SequenceDecoder`] that
//! owns it, so no thread is spawned per frame. Jobs are `'static`
//! closures executed in FIFO order across all submitted batches;
//! [`WorkerPool::submit`] queues a frame's batch without blocking and
//! [`WorkerPool::wait`] blocks until a batch has finished — the calling
//! thread works the queue meanwhile.
//!
//! Jobs may block on progress published by jobs queued *earlier* (the
//! frame pipeline's loop-filter rows wait on tile rows and on the row
//! above; a frame's tile rows wait on the reference rows of the frames
//! before it). FIFO dequeueing makes that deadlock-free at any pool
//! size: when a job is running, every job queued before it has been
//! dequeued, so the earliest unfinished job is always running and only
//! ever waits on finished work. With no helper threads the caller runs
//! the queue in order.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

type Panic = Box<dyn std::any::Any + Send + 'static>;

/// One submitted batch's completion state.
struct BatchState {
    /// (jobs not yet finished, first panic payload).
    pending: Mutex<(usize, Option<Panic>)>,
    done: Condvar,
}

/// A handle on a submitted batch (see [`WorkerPool::wait`]).
pub(crate) struct Batch {
    state: Arc<BatchState>,
}

impl Batch {
    /// Whether every job of the batch has finished.
    #[cfg(test)]
    pub(crate) fn is_done(&self) -> bool {
        lock(&self.state.pending).0 == 0
    }
}

struct Queue {
    /// `(job, its batch, its index inside the batch)`.
    jobs: VecDeque<(Job, Arc<BatchState>, usize)>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when a job is queued (helpers wait on it).
    work: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run one job, recording a panic in its batch instead of unwinding.
fn execute(job: Job, batch: &BatchState) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    let mut p = lock(&batch.pending);
    if let Err(payload) = result
        && p.1.is_none()
    {
        p.1 = Some(payload);
    }
    p.0 -= 1;
    if p.0 == 0 {
        batch.done.notify_all();
    }
}

/// The persistent pool: `helpers` background threads plus the calling
/// thread while it is inside [`WorkerPool::wait`].
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
    /// every job itself, inside [`Self::wait`]).
    pub(crate) fn new(helpers: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                shutdown: false,
            }),
            work: Condvar::new(),
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

    /// Queue `jobs` (FIFO, after every job already queued).
    pub(crate) fn submit(&self, jobs: Vec<Job>) -> Batch {
        let state = Arc::new(BatchState {
            pending: Mutex::new((jobs.len(), None)),
            done: Condvar::new(),
        });
        {
            let mut q = lock(&self.shared.queue);
            q.jobs.extend(
                jobs.into_iter()
                    .enumerate()
                    .map(|(i, j)| (j, Arc::clone(&state), i)),
            );
        }
        self.shared.work.notify_all();
        Batch { state }
    }

    /// Block until `batch` has finished, running queued jobs (of any
    /// batch, in FIFO order) meanwhile. A panicking job's payload is
    /// re-raised here.
    pub(crate) fn wait(&self, batch: &Batch) {
        loop {
            if lock(&batch.state.pending).0 == 0 {
                break;
            }
            let job = lock(&self.shared.queue).jobs.pop_front();
            match job {
                Some((job, state, _)) => execute(job, &state),
                None => {
                    // Everything left is running on helpers.
                    let mut p = lock(&batch.state.pending);
                    while p.0 > 0 {
                        p = batch.state.done.wait(p).unwrap_or_else(|e| e.into_inner());
                    }
                    break;
                }
            }
        }
        let payload = lock(&batch.state.pending).1.take();
        if let Some(payload) = payload {
            std::panic::resume_unwind(payload);
        }
    }

    /// Run, on the calling thread, the first still-queued job of `batch`
    /// whose index inside the batch is below `below`; `false` when there
    /// is none (each is running or finished).
    ///
    /// This takes a job out of FIFO order, so it is only for jobs that
    /// block on nothing but jobs queued earlier *which the helpers will
    /// reach on their own*: the helpers' FIFO argument is untouched (the
    /// job is simply gone from their queue, and nothing queued before it
    /// waits on it), and the caller's wait ends because the helpers keep
    /// making progress. With no helper there is no out-of-order hazard
    /// either: the caller only gets here between its own FIFO runs.
    pub(crate) fn run_queued(&self, batch: &Batch, below: usize) -> bool {
        let job = {
            let mut q = lock(&self.shared.queue);
            let at = q
                .jobs
                .iter()
                .position(|(_, state, index)| Arc::ptr_eq(state, &batch.state) && *index < below);
            at.and_then(|at| q.jobs.remove(at))
        };
        match job {
            Some((job, state, _)) => {
                execute(job, &state);
                true
            }
            None => false,
        }
    }

    /// [`Self::submit`] + [`Self::wait`].
    #[cfg(test)]
    pub(crate) fn run(&self, jobs: Vec<Job>) {
        let batch = self.submit(jobs);
        self.wait(&batch);
    }
}

fn helper_loop(shared: &Shared) {
    loop {
        let (job, state) = {
            let mut q = lock(&shared.queue);
            loop {
                if let Some((job, state, _)) = q.jobs.pop_front() {
                    break (job, state);
                }
                if q.shutdown {
                    return;
                }
                q = shared.work.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        execute(job, &state);
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        lock(&self.shared.queue).shutdown = true;
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

    /// Jobs waiting on earlier jobs — of the same or an earlier batch —
    /// never deadlock, even with fewer threads than jobs.
    #[test]
    fn chained_waits_complete_with_one_thread() {
        for helpers in [0, 1, 2] {
            let pool = WorkerPool::new(helpers);
            let state = Arc::new((Mutex::new(0usize), Condvar::new()));
            let make = |range: std::ops::Range<usize>| -> Vec<Job> {
                range
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
                    .collect()
            };
            let first = pool.submit(make(0..6));
            let second = pool.submit(make(6..12));
            pool.wait(&second);
            pool.wait(&first);
            assert!(first.is_done() && second.is_done());
            assert_eq!(*state.0.lock().unwrap(), 12);
        }
    }

    /// The caller can run the leading jobs of a batch itself while the
    /// helpers are held by an earlier batch; the batch's later jobs stay
    /// queued for the FIFO order.
    #[test]
    fn caller_runs_the_leading_jobs_of_a_batch_out_of_order() {
        let pool = WorkerPool::new(1);
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let blocker = {
            let gate = Arc::clone(&gate);
            pool.submit(vec![Box::new(move || {
                let (m, cv) = &*gate;
                let mut open = m.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
            }) as Job])
        };
        let ran = Arc::new(Mutex::new(Vec::new()));
        let jobs: Vec<Job> = (0..4)
            .map(|i| {
                let ran = Arc::clone(&ran);
                Box::new(move || ran.lock().unwrap().push(i)) as Job
            })
            .collect();
        let batch = pool.submit(jobs);
        // The only helper is (or will be) inside the blocker: jobs 0 and
        // 1 run here, in order; 2 and 3 are not eligible.
        assert!(pool.run_queued(&batch, 2));
        assert!(pool.run_queued(&batch, 2));
        assert!(!pool.run_queued(&batch, 2));
        assert_eq!(*ran.lock().unwrap(), vec![0, 1]);
        assert!(!pool.run_queued(&blocker, 0));
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        pool.wait(&batch);
        pool.wait(&blocker);
        assert_eq!(*ran.lock().unwrap(), vec![0, 1, 2, 3]);
        // Nothing left to take.
        assert!(!pool.run_queued(&batch, 4));
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
