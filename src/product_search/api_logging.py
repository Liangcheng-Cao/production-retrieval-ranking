"""Small JSON logger and request-ID/HTTP timing middleware; no query text logs."""
import json
import logging
import re
from time import perf_counter
from uuid import uuid4

LOGGER = logging.getLogger('product_search.service')


def event(name, **fields):
    LOGGER.info(json.dumps({'event': name, **fields}, ensure_ascii=True, allow_nan=False))


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        start = perf_counter()
        incoming = [v for k, v in scope.get('headers', []) if k.lower() == b'x-request-id']
        request_id = incoming[0].decode('ascii', errors='ignore') if len(incoming) == 1 else ''
        if (not incoming or len(incoming) != 1 or incoming[0] != request_id.encode('ascii')
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', request_id)):
            request_id = str(uuid4())
        state = scope.setdefault('state', {})
        state['request_id'] = request_id
        search = scope.get('path') == '/search' and scope.get('method') == 'POST'
        metrics = scope['app'].state.metrics if search else None
        if metrics:
            metrics.inflight.inc()
        status = 499
        started = False
        async def send_timed(message):
            nonlocal started, status
            if message['type'] == 'http.response.start':
                started = True
                status = message['status']
                http_ms = (perf_counter()-start)*1000
                message['headers'] = list(message.get('headers', [])) + [
                    (b'x-request-id', request_id.encode('ascii')),
                    (b'x-http-app-ms', f'{http_ms:.6f}'.encode())]
                if 'engine_ms' in state:
                    message['headers'].append((b'x-engine-ms', f"{state['engine_ms']:.6f}".encode()))
                if 'serialization_ms' in state:
                    message['headers'].append((b'x-response-serialization-ms', f"{state['serialization_ms']:.6f}".encode()))
                if 'queue_wait_ms' in state:
                    message['headers'].append((b'x-queue-wait-ms', f"{state['queue_wait_ms']:.6f}".encode()))
                if scope.get('path') == '/search':
                    event('http_response', request_id=request_id, status=message['status'], http_app_ms=http_ms,
                          engine_total_ms=state.get('engine_ms'), serialization_ms=state.get('serialization_ms'))
            await send(message)
        try:
            await self.app(scope, receive, send_timed)
        except Exception:
            event('request_failed', request_id=request_id, reason='internal_error')
            if started:
                raise
            from starlette.responses import JSONResponse
            response = JSONResponse({'error': 'internal_error', 'message': 'Request could not be completed.',
                                     'request_id': request_id}, status_code=500)
            await response(scope, receive, send_timed)
        finally:
            if metrics:
                seconds = perf_counter()-start
                metrics.inflight.dec()
                metrics.finish(state, status, seconds)
                name = ('search_fallback' if state.get('fallback_used') else 'search_completed') if status == 200 else 'search_failed'
                event(name, request_id=request_id, status=status, http_total_ms=seconds*1000,
                      **{k: state.get(k) for k in ('pipeline', 'effective_pipeline', 'top_k', 'query_length',
                         'queue_wait_ms', 'fallback_used', 'result_count')}, engine_total_ms=state.get('engine_ms'))
