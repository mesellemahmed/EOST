# EOST v2 scientific runtime (CPU only).
# All scientific execution (data building, calibration, adaptive replay,
# statistics, figures) happens inside this image. The host only orchestrates
# Docker ("Host = Docker orchestration only. Science = container only.").
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    XLA_FLAGS=--xla_force_host_platform_device_count=1 \
    JAX_PLATFORMS=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Pinned, fully resolved dependency set (see requirements.in / `./run lock`).
# An optional extra CA certificate can be supplied as a BuildKit secret
# (id=extra_ca) when building behind a TLS-intercepting proxy; it is only
# used during this RUN step and never stored in the image.
COPY requirements.lock /tmp/requirements.lock
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /work
ENV PYTHONPATH=/work/src
CMD ["python", "-m", "eost.doctor"]
