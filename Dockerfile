FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache openenv-core>=0.2.0 fastapi uvicorn[standard] pydantic openai requests pyyaml

COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir -e . --no-deps

USER appuser

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
