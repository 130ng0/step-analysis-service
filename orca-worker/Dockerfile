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
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/orca

RUN wget -O OrcaSlicer.AppImage \
    https://github.com/OrcaSlicer/OrcaSlicer/releases/download/v2.3.1/OrcaSlicer_Linux_AppImage_V2.3.1.AppImage

RUN chmod +x OrcaSlicer.AppImage
RUN ./OrcaSlicer.AppImage --appimage-extract

COPY orca-worker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]