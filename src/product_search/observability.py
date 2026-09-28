"""Per-service, bounded-cardinality aggregates; no query/result retention."""
from threading import Lock
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

PIPELINES = ('bm25', 'hybrid', 'hybrid_rerank')
STAGES = ('bm25', 'dense_encoding', 'dense_search', 'fusion', 'reranking', 'hydration')
SECONDS = (.0001, .0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5)


def mode(value):
    return value if value in PIPELINES else 'unknown'


class ServiceMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        labels = ('pipeline', 'effective_pipeline', 'status')
        def counter(name, help):
            return Counter(name, help, labels, registry=self.registry)
        self.requests = counter('request_count', 'POST /search responses (or cancelled requests).')
        self.errors = counter('request_error_count', 'Non-2xx search outcomes, including cancellation 499.')
        self.fallbacks = counter('fallback_count', 'Search responses using reranker fallback.')
        self.reranker_fallbacks = counter('reranker_fallback_total', 'Alias of fallback_count_total.')
        self.degraded = counter('degraded_request_total', 'Requests admitted or completed in DEGRADED, or with fallback.')
        self.empty = counter('empty_result_total', 'Successful empty search responses.')
        self.http = Histogram('request_latency_seconds', 'ASGI search lifetime including response body; not client RTT.', labels, buckets=SECONDS, registry=self.registry)
        self.engine = Histogram('engine_latency_seconds', 'Core total_ms on returned results, unchanged semantics.', ('pipeline',), buckets=SECONDS, registry=self.registry)
        self.wait = Histogram('queue_wait_seconds', 'Admission to actual executor callable entry; excludes cancelled-before-start.', ('pipeline',), buckets=SECONDS, registry=self.registry)
        self.results = Histogram('result_count', 'Number of hits on successful search responses.', labels, buckets=(0, 1, 5, 10, 20, 50, 100), registry=self.registry)
        self.chars = Histogram('query_character_length', 'Unicode codepoints of validated API query.', ('pipeline',), buckets=(0, 4, 8, 16, 32, 64, 128, 256, 512), registry=self.registry)
        self.tokens = Histogram('query_token_count', 'Whitespace-delimited tokens of validated API query; not model tokens.', ('pipeline',), buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 512), registry=self.registry)
        self.stages = {s: Histogram(s+'_latency_seconds', 'Core '+s+'_ms for returned results, including zero for unused stage.', ('pipeline',), buckets=SECONDS, registry=self.registry) for s in STAGES}
        self.inflight = Gauge('in_flight_requests', 'Active POST /search ASGI calls until completion/cancellation.', registry=self.registry)
        self.waiting = Gauge('requests_waiting_for_engine', 'Live admitted searches not yet entering the engine executor callable.', registry=self.registry)
        self.active = Gauge('engine_active_calls', 'Actual executing search callables, including disconnected callers.', registry=self.registry)

    def query(self, pipeline, text):
        self.chars.labels(mode(pipeline)).observe(len(text))
        self.tokens.labels(mode(pipeline)).observe(len(text.split()))

    def core_result(self, result):
        pipeline = mode(result.requested_pipeline)
        self.engine.labels(pipeline).observe(result.timing['total_ms']/1000)
        for stage in STAGES:
            self.stages[stage].labels(pipeline).observe(result.timing[stage+'_ms']/1000)

    def finish(self, state, status, seconds):
        labels = (mode(state.get('pipeline')), mode(state.get('effective_pipeline')),
                  str(status) if status in (200, 400, 404, 405, 413, 422, 499, 500, 503) else 'other')
        self.requests.labels(*labels).inc()
        self.http.labels(*labels).observe(seconds)
        if not 200 <= status < 300:
            self.errors.labels(*labels).inc()
        if state.get('degraded') or state.get('fallback_used'):
            self.degraded.labels(*labels).inc()
        if state.get('fallback_used'):
            self.fallbacks.labels(*labels).inc()
            self.reranker_fallbacks.labels(*labels).inc()
        if status == 200 and 'result_count' in state:
            self.results.labels(*labels).observe(state['result_count'])
            if state['result_count'] == 0:
                self.empty.labels(*labels).inc()

    def render(self):
        return generate_latest(self.registry)


class WaitingTicket:
    """Thread-safe admission accounting across event loop, executor, cancellation."""
    def __init__(self, metrics, pipeline):
        self.metrics, self.pipeline = metrics, mode(pipeline)
        self.start = perf_counter()
        self.pending = True
        self.lock = Lock()
        metrics.waiting.inc()

    def enter(self):
        elapsed = perf_counter()-self.start
        with self.lock:
            if self.pending:
                self.metrics.waiting.dec()
                self.metrics.wait.labels(self.pipeline).observe(elapsed)
                self.pending = False
        return elapsed*1000

    def cancel(self):
        with self.lock:
            if self.pending:
                self.metrics.waiting.dec()
                self.pending = False
