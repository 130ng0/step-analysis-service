FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    file \
    xz-utils \
    python3 \
    python3-pip \
    libfuse2 \
    libgl1 \
    libegl1 \
    libopengl0 \
    libglu1-mesa \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgtk-3-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libsoup-3.0-0 \
    libjavascriptcoregtk-4.1-0 \
    libwebkit2gtk-4.1-0 \
    libmspack0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    fastapi uvicorn python-multipart \
    trimesh numpy networkx lxml

WORKDIR /opt/orca

ARG ORCA_SLICER_VERSION=2.5.2
ARG ORCA_SLICER_SHA256=""

RUN set -eu; \
    printf '%s\n' "$ORCA_SLICER_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; \
    checksum="$ORCA_SLICER_SHA256"; \
    if [ -z "$checksum" ]; then \
        case "$ORCA_SLICER_VERSION" in \
            2.5.2) checksum=22fa88884dec3fdbd29e90cba9912fcff6fb57a60b1bdc1a6c189c61a6b0a870 ;; \
            2.5.1) checksum=073b8105c928356141db0ddbce88b81d52c7a9e2442f9baae0a9089c89ff04eb ;; \
            *) echo "Set ORCA_SLICER_SHA256 to the official digest for OrcaSlicer $ORCA_SLICER_VERSION" >&2; exit 1 ;; \
        esac; \
    fi; \
    printf '%s\n' "$checksum" | grep -Eq '^[[:xdigit:]]{64}$'; \
    base_url="https://github.com/OrcaSlicer/OrcaSlicer/releases/download/v${ORCA_SLICER_VERSION}"; \
    asset="OrcaSlicer_Linux_AppImage_Ubuntu2404_V${ORCA_SLICER_VERSION}.AppImage"; \
    wget -O OrcaSlicer.AppImage "$base_url/$asset"; \
    printf '%s  OrcaSlicer.AppImage\n' "$checksum" | sha256sum -c -; \
    chmod +x OrcaSlicer.AppImage; \
    ./OrcaSlicer.AppImage --appimage-extract > /dev/null

ENV ORCA_SLICER_VERSION=${ORCA_SLICER_VERSION}

WORKDIR /workspace

COPY orca-worker/api.py /workspace/api.py
COPY orca-worker/profile_loader.py /workspace/profile_loader.py
COPY orca-worker/entrypoint.sh /entrypoint.sh
COPY orca-worker/orca-profiles /seed/profiles
COPY orca-worker/stl-test /seed/stl-test
COPY orca-worker/resolve_profiles.py /workspace/resolve_profiles.py
COPY orca-worker/orientation.py /workspace/orientation.py

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
