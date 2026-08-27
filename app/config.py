import os

API_KEY = os.getenv("STEP_SERVICE_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = (".step", ".stp", ".stl")

# Number of jobs processed in parallel by the analysis service.
# Keep the hard limit conservative because each active job may launch its own
# OrcaSlicer process in the downstream worker and can consume significant CPU/RAM.
ANALYSIS_MAX_WORKERS_HARD_LIMIT = 10


def _analysis_max_workers() -> int:
    raw_value = os.getenv("ANALYSIS_MAX_WORKERS", "5")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 3

    return max(1, min(value, ANALYSIS_MAX_WORKERS_HARD_LIMIT))


ANALYSIS_MAX_WORKERS = _analysis_max_workers()
