FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libfuse2 \
    file \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/orca

RUN wget -O OrcaSlicer.AppImage \
    https://github.com/OrcaSlicer/OrcaSlicer/releases/latest/download/OrcaSlicer_Linux.AppImage

RUN chmod +x OrcaSlicer.AppImage
RUN ./OrcaSlicer.AppImage --appimage-extract

COPY orca-worker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]