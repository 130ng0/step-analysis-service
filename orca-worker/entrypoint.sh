#!/bin/bash
set -e

echo "[orca-worker] container started"

mkdir -p /workspace/profiles
mkdir -p /workspace/stl-test
mkdir -p /workspace/out

cp -rn /seed/profiles/* /workspace/profiles/ 2>/dev/null || true
cp -rn /seed/stl-test/* /workspace/stl-test/ 2>/dev/null || true

echo "[orca-worker] profiles in /workspace/profiles:"
ls -1 /workspace/profiles || true

echo "[orca-worker] stl files in /workspace/stl-test:"
ls -1 /workspace/stl-test || true

if [ "$1" = "help" ]; then
    /opt/orca/squashfs-root/AppRun --help || true
    exit 0
fi

if [ "$1" = "bash" ]; then
    exec /bin/bash
fi

tail -f /dev/null