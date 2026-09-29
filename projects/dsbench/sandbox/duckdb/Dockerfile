# Model-written DuckDB SQL runs here, never in a host process (ADR-004 Revision 2): DuckDB SQL can
# read and write files and fetch URLs. The compose service gives this container no network, a
# read-only root filesystem and no host mounts, so a query sees only the table it is sent.
# The DuckDB version matches the one uv.lock pins for the host, so gold SQL checked in-process and
# model SQL checked here run on the same engine.
FROM python:3.12-slim
RUN pip install --no-cache-dir duckdb==1.5.5
COPY runner.py /runner.py
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 65534:65534
CMD ["sleep", "infinity"]
