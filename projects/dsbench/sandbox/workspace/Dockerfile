# The agent's hands: `run_python` execs inside this container (never the host). ADR-003 §1.
# Lean for Phase 1 (warehouse work). Airflow client + heavier deps arrive in Phase 2.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        clickhouse-connect==0.8.* \
        pandas==2.* \
        polars==1.* \
        numpy==2.* \
        scipy==1.* \
        scikit-learn==1.* \
        pyarrow==17.* \
        mlflow==2.* \
        httpx==0.27.* \
        requests==2.*

WORKDIR /repo
