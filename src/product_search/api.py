"""Thin HTTP adapter. Engine construction only occurs inside lifespan."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from .api_logging import RequestContextMiddleware, event, LOGGER
from .api_schemas import (SearchRequest, SearchResponse, ErrorResponse, HealthResponse,
                          ReadyResponse, PublicVersion, public_version)
from .runtime_config import RuntimeConfig
from .search_engine import SearchEngine
from .observability import ServiceMetrics, WaitingTicket


def create_app(config_path=None, *, engine_factory=None):
    # Factory injection is for tests, never an HTTP-accessible control.
    @asynccontextmanager
    async def lifespan(app):
        app.state.metrics = ServiceMetrics()
        event('service_starting')
        app.state.engine = None
        app.state.phase = 'LOADING'
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='search-core')
        app.state.search_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        def load():
            path = config_path or os.environ.get('SEARCH_RUNTIME_CONFIG', 'configs/runtime.json')
            engine = engine_factory() if engine_factory else SearchEngine(RuntimeConfig.load(Path(path)))
            app.state.engine = engine
            engine.initialize()
            return public_version(engine.version)
        try:
            event('engine_loading')
            app.state.public_version = await loop.run_in_executor(app.state.executor, load)
            app.state.phase = 'READY'
            event('service_ready', runtime_manifest_sha256=app.state.public_version.runtime_manifest_sha256)
            yield
        except Exception:
            app.state.phase = 'FAILED'
            LOGGER.exception('service_lifecycle_failed')
            raise
        finally:
            try:
                if app.state.engine is not None:
                    await loop.run_in_executor(app.state.executor, app.state.engine.close)
            finally:
                app.state.executor.shutdown(wait=True)
                app.state.phase = 'CLOSED'
                event('service_shutdown')

    app = FastAPI(title='Product Search', version='1.0.0', lifespan=lifespan)
    app.state.engine = None
    app.state.phase = 'NOT_INITIALIZED'
    app.state.metrics = ServiceMetrics()
    app.add_middleware(RequestContextMiddleware)

    def error(request, status, code, message):
        return JSONResponse(ErrorResponse(error=code, message=message,
                            request_id=request.state.request_id).model_dump(), status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        event('search_rejected', request_id=request.state.request_id, reason='invalid_request')
        # Pydantic errors can contain raw input; deliberately don't echo/log them.
        return error(request, 422, 'invalid_request', 'Invalid JSON or search fields; see OpenAPI schema.')

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(request, exc.status_code, 'http_error', 'HTTP request could not be handled.')

    def status():
        engine = app.state.engine
        if engine is None:
            return ReadyResponse(ready=False, searchable=False, state=app.state.phase,
                                 enabled_pipelines=[], detail='Engine unavailable')
        state = engine.readiness()
        return ReadyResponse(ready=state.ready, searchable=state.state in ('READY', 'DEGRADED'),
            state=state.state, enabled_pipelines=list(state.enabled_pipelines),
            detail='Component degradation; fallback may apply' if state.state == 'DEGRADED' else
                   None if state.ready else 'Engine unavailable')

    errors = {n: {'model': ErrorResponse} for n in (422, 500, 503)}

    @app.get('/metrics', include_in_schema=False)
    async def metrics():
        return Response(app.state.metrics.render(), media_type='text/plain; version=0.0.4; charset=utf-8')

    @app.get('/health', response_model=HealthResponse)
    async def health():
        return HealthResponse(alive=True)

    @app.get('/ready', response_model=ReadyResponse, responses={503: {'model': ReadyResponse}})
    async def ready():
        value = status()
        return JSONResponse(value.model_dump(), status_code=200 if value.searchable else 503)

    @app.get('/version', response_model=PublicVersion, responses={503: {'model': ErrorResponse}})
    async def version(request: Request):
        if not status().searchable:
            return error(request, 503, 'not_ready', 'Search engine is unavailable.')
        return app.state.public_version

    @app.post('/search', response_model=SearchResponse, responses=errors)
    async def search(body: SearchRequest, request: Request):
        common = {'request_id': request.state.request_id, 'pipeline': body.pipeline,
                  'top_k': body.top_k, 'query_length': len(body.query)}
        request.state.pipeline = body.pipeline
        request.state.top_k = body.top_k
        request.state.query_length = len(body.query)
        app.state.metrics.query(body.pipeline, body.query)
        current = status()
        request.state.degraded = current.state == 'DEGRADED'
        if not current.searchable:
            event('search_failed', **common, reason='not_ready')
            return error(request, 503, 'not_ready', 'Search engine is unavailable.')
        if body.pipeline not in app.state.engine.config.enabled_pipelines:
            event('search_rejected', **common, reason='disabled_pipeline')
            return error(request, 422, 'invalid_request', 'Requested pipeline is disabled.')
        try:
            # A single worker preserves the sequential core's ownership contract.
            # Cancellation/disconnect cannot preempt native/GPU work; no fake timeout.
            ticket = WaitingTicket(app.state.metrics, body.pipeline)
            def execute():
                request.state.queue_wait_ms = ticket.enter()
                app.state.metrics.active.inc()
                try:
                    value = app.state.engine.search(body.query, body.top_k, body.pipeline)
                    app.state.metrics.core_result(value)
                    return value
                finally:
                    request.state.degraded |= app.state.engine.readiness().state == 'DEGRADED'
                    app.state.metrics.active.dec()
            try:
                async with app.state.search_lock:
                    result = await asyncio.get_running_loop().run_in_executor(app.state.executor, execute)
            finally:
                ticket.cancel()
            request.state.engine_ms = result.timing['total_ms']
            t = perf_counter()
            value = SearchResponse(request_id=request.state.request_id, query=result.query,
                requested_pipeline=result.requested_pipeline, effective_pipeline=result.effective_pipeline,
                fallback_used=result.fallback_used,
                fallback_reason='reranker_unavailable' if result.fallback_used else None,
                results=[asdict(hit) for hit in result.results], timing_ms=dict(result.timing),
                version=app.state.public_version)
            encoded = value.model_dump_json()
            request.state.serialization_ms = (perf_counter()-t)*1000
            request.state.effective_pipeline = result.effective_pipeline
            request.state.fallback_used = result.fallback_used
            request.state.result_count = len(result.results)
            return Response(encoded, media_type='application/json')
        except Exception:
            # Exception messages/tracebacks may contain queries; never log them by default.
            event('search_failed', **common, reason='core_error')
            return error(request, 500, 'search_failed', 'Search could not be completed.')

    return app


app = create_app()
