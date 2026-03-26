#!/bin/bash
set -e

echo "[orca-worker] container started"

if [ "$1" = "help" ]; then
    /opt/orca/squashfs-root/AppRun --help || true
    exit 0
fi

if [ "$1" = "bash" ]; then
    exec /bin/bash
fi

tail -f /dev/null