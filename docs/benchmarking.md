# Reproducible latency and closed-loop load measurements

Phase 6 measures the frozen SearchEngine and existing HTTP adapter. It changes no
model, ranking parameter, core contract or HTTP dispatch policy. The tested service
still has one Uvicorn process and one engine owner thread. No deployment, autoscaling,
queue/admission implementation, hard inference timeout or SLA is introduced.

## Protocol

The versioned protocol is configs/phase6_plan.json. Select the first 24 sorted train
query IDs from the frozen manifest; no relevance labels are needed. Top-K=10 and
all three modes are fixed. Each case uses an identical seed-42 shuffled balanced
schedule, so every query appears equally often. All actual requests invoke the
engine; there is no query-result cache.

Two fresh service processes provide independent repetitions. Each run:

1. Pre-import native ML dependencies on the main thread, before limiting BLAS and
   creating the engine owner thread. This loads libraries, not model weights/indexes.
2. Initialize the fully enabled engine once; record its own startup timings and
   process/GPU memory snapshot. Imports/process boot are outside this startup measure.
3. Build small direct-reference result fixtures; warm each mode eight times, then
   collect 960 sequential direct-core observations per mode, including stage timings.
4. Expose the unchanged HTTP app; deterministically shuffle the twelve combinations
   of mode × concurrency (1, 2, 4, 8), independently for each run.
5. For each case, warm eight sequential requests, then issue 960 measured requests
   through C closed-loop client workers with no think time and reused HTTP connections.
6. Request graceful shutdown over the child's stdin, verify CLOSED, and retain the
   access-boundary log. No public shutdown/fault endpoint is added.

The client and server are separate Python processes but share one Windows host and
loopback network. Torch and BLAS use one thread; deterministic algorithms are on,
TF32 is off. Model inference stays on the configured RTX 4070. Service JSON INFO
events are written to an ignored local log; access logs are disabled. The client
validates each response before issuing its next request, so client processing is
part of the closed-loop throughput context.

## Timing and accounting

- Direct wall time covers SearchEngine.search; the core's stage timers are retained.
- Client latency covers HTTP request send through receiving the body. JSON/parity
  checking occurs afterward and affects the next request's issue time.
- HTTP application duration and serialization duration come from existing headers.
- Engine duration comes from the existing response timing object.
- Server non-engine duration is HTTP application minus engine time. It includes
  scheduling/waiting, validation, serialization and other adapter work. It is **not**
  a direct or pure queue-wait measurement.

Report P50/P95/P99/maximum/mean with sample count. Percentiles use NumPy's default
linear interpolation. Never average P95s and label the result a pooled percentile;
the human report presents the two run values as ranges. Successful throughput is
successful completed requests divided by measured case wall time. All attempts,
errors, exception types, fallbacks and result mismatches are retained; failed requests
are not silently omitted from all-attempt latency or error counts.

Every 200 response is checked against direct-core expected IDs/order and final scores
(absolute/relative tolerance 1e-6 for BM25/Hybrid, 1e-4 for CE), plus effective mode
and fallback state. This verifies result stability under the tested concurrency.
It does not evaluate relevance or add claims about final-test ranking quality.

## Scope and limitations

This is a bounded closed-loop workload, not an open-loop arrival-rate test. It cannot
establish behavior under an independently imposed overload rate, and it has the
usual closed-loop/coordinated-omission limitation. C=8 is the tested maximum, not a
capacity claim. Waiting is not admission-bounded in the current application; this
benchmark adds no queue or request cancellation. The client uses a 30-second
operation timeout; that is not a hard server inference deadline.

The single engine thread serializes model work. Increasing outstanding clients may
raise throughput by reducing idle gaps, then mostly raises waiting and tail latency.
The generator, JSON/logging overhead and shared host can limit measured BM25 QPS.
No claim of CPU/GPU saturation follows solely from these measurements. Other desktop
workloads, thermal state and power settings are uncontrolled. Two repetitions show
variation but are insufficient for population-level confidence intervals or SLA claims.

The corpus/index/model pages and model weights may already be cached by the OS;
fresh process startup is not a cold-disk/download test. GPU numerical equivalence
is limited to the fixed runtime, retaining the Phase 4 batch-context caveat.

## Preserved preliminary evidence

Two initial startup attempts stalled while SciPy's native BLAS extension was first
imported on the owner worker after the main thread had applied threadpool limits.
The captured thread stack identifies the import location; it does not prove the
underlying native-library cause. Those owned processes were explicitly stopped and
their failed reports/logs retained. Main-thread dependency pre-import resolved the
observed harness startup problem. Serving and ranking code were not modified.

The subsequent 96-request pilot completed without errors. BM25 cases lasted less
than a second and showed noticeable variation. Before formal measurements, the
protocol was revised uniformly to 960 requests for every case and both runs.
Pilot reports retain their original embedded plans/hashes and are not pooled with
the formal results. No run was selected for being faster.

## Reproduction and evidence

From the repository, with the existing runtime artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_benchmark.py --report reports/tmp/phase6-repeat-summary.json --raw-root reports/tmp/phase6-repeat
```

Both paths must be new: the harness refuses overwrite. Raw observations and logs
must stay under ignored reports/tmp. Compact versioned reports include protocol,
hardware/software, artifact identity, script hashes, per-run summaries and raw-file
SHA256 references. Keep the ignored raw directories locally for audit/reanalysis;
Git contains their checksums and compact summaries, not large observation dumps.

Final-test labels are never opened. Service boundary guards permit product metadata
and train query text while blocking raw data, actual test judgments and the global
conflict audit. The client does not read any relevance labels.
