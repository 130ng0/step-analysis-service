#!/bin/bash
set -e

echo "[orca-worker] container started"

mkdir -p /workspace/profiles
mkdir -p /workspace/stl-test
mkdir -p /workspace/templates
mkdir -p /workspace/out

cp -rn /seed/profiles/* /workspace/profiles/ 2>/dev/null || true
cp -rn /seed/stl-test/* /workspace/stl-test/ 2>/dev/null || true
cp -rn /seed/templates/* /workspace/templates/ 2>/dev/null || true

echo "[orca-worker] profiles in /workspace/profiles:"
ls -1 /workspace/profiles || true

echo "[orca-worker] stl files in /workspace/stl-test:"
ls -1 /workspace/stl-test || true

echo "[orca-worker] templates in /workspace/templates:"
ls -1 /workspace/templates || true

exec uvicorn api:app --host 0.0.0.0 --port 8090