# Batch image: ingestion CronJob, index backfill Job, and the eval gate.
#
# Separate from the two serving images because it needs things they must not have --
# git (to clone the docs), torch (to embed), and the training/eval code. Folding these
# into the API image would drag all of it into the request path's blast radius.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[ml,serve]"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data /opt/models \
    && chown -R app:app /data /opt/models /app

ENV KVRAG_DATA_DIR=/data \
    PYTHONPATH=/app/src

USER app

# No default command: this image is invoked with an explicit argv by each Job and
# CronJob. A default here would make a misconfigured workload silently do something.
ENTRYPOINT ["python", "-m"]
