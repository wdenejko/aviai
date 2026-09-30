# SQL on SynSQL's SQLite databases runs here, never in a host process (ADR-004 Revision 2): the
# dataset's gold SQL and the model's both. The compose service gives this container no network, a
# read-only root filesystem and no host mounts; the runner opens each database read-only.
FROM python:3.12-slim
COPY runner.py /runner.py
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 65534:65534
CMD ["sleep", "infinity"]
