# Target: Linux amd64, CPython 3.14, existing CUDA 13.0 torch runtime.
# Artifact/model payloads are deliberately absent from all image layers.
FROM python:3.14.4-slim-trixie@sha256:2409290aa375de35f6492db84c700067d5c4c2aacfaf770c155d7528fb68bcf1 AS builder
WORKDIR /build
COPY deployment/linux-cp314-cu130.lock /build/runtime.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --require-hashes --no-deps -r runtime.lock
COPY pyproject.toml /build/pyproject.toml
COPY src /build/src
ARG SOURCE_DATE_EPOCH=1790599546
ENV SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH
RUN /opt/venv/bin/python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheel . \
    && /opt/venv/bin/python -m pip install --no-deps /wheel/*.whl \
    && /opt/venv/bin/python -m pip check

FROM python:3.14.4-slim-trixie@sha256:2409290aa375de35f6492db84c700067d5c4c2aacfaf770c155d7528fb68bcf1
# Debian libraries are resolved from the base distribution repositories at build
# time; Python wheels/base image are hash-pinned, OS rebuilds are not bit-identical.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY deployment/healthcheck.py /app/healthcheck.py
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    SEARCH_RUNTIME_CONFIG=/runtime/runtime.json
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "product_search.deployment"]
