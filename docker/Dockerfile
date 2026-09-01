# syntax=docker/dockerfile:1.9

# ReconVision runtime image.
#
# Deliberately does not contain PyTorch. Model export is a development-time step
# that produces ONNX files (see ADR 0002), so the runtime needs only ONNX Runtime -
# which is the difference between an image of a few hundred megabytes and one of
# nearly three gigabytes on a home NAS.

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained virtualenv.
# ---------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Built at the path the venv will live at in the final image. Console scripts
# embed an absolute shebang, so a venv built in /build and copied to /app points
# at an interpreter that no longer exists - and the container fails with an
# "exec: no such file or directory" that names the script rather than the cause.
WORKDIR /app

# Dependencies first, sources second: editing application code must not invalidate
# the layer holding several hundred megabytes of wheels.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra runtime --extra api --extra notify --extra telemetry

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable \
        --extra runtime --extra api --extra notify --extra telemetry

# ---------------------------------------------------------------------------
# Runtime.
# ---------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS runtime

# OpenCV headless still needs the shared libraries behind image and video codecs.
# libgl and the X11 stack are deliberately absent: nothing here draws to a screen.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# An unprivileged user, and one that owns the data volume: this process handles
# camera footage and face descriptors, and has no reason to be root.
RUN groupadd --gid 1000 reconvision \
    && useradd --uid 1000 --gid 1000 --create-home reconvision

COPY --from=builder --chown=reconvision:reconvision /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECONVISION_DATA_DIR=/data

WORKDIR /app
RUN mkdir -p /data && chown reconvision:reconvision /data
VOLUME ["/data"]

USER reconvision
EXPOSE 8000

# Checks the application, not the port: a process that has lost its database
# still accepts connections, and reporting that as healthy is worse than useless.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["reconvision"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
