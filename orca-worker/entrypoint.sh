#!/bin/bash
set -e

echo "[orca-worker] container started"

mkdir -p /workspace/profiles
mkdir -p /workspace/stl-test
mkdir -p /workspace/out

# Workspace-Inhalte bewusst aktualisieren
rm -rf /workspace/profiles/*
rm -rf /workspace/stl-test/*

cp -r /seed/profiles/. /workspace/profiles/
cp -r /seed/stl-test/. /workspace/stl-test/

echo "[orca-worker] profiles in /workspace/profiles:"
find /workspace/profiles -maxdepth 3 -type f | sort || true

echo "[orca-worker] stl files in /workspace/stl-test:"
find /workspace/stl-test -maxdepth 2 -type f | sort || true


ORCA_WORKER_COUNT="${ORCA_WORKER_COUNT:-2}"
echo "[orca-worker] uvicorn workers: ${ORCA_WORKER_COUNT}"
exec uvicorn api:app --host 0.0.0.0 --port 8090 --workers "${ORCA_WORKER_COUNT}"