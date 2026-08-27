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


# Terminal job metadata/result is retained briefly so repeated Odoo polls can
# read the same result safely. Binary and temporary job files are deleted
# immediately after each job finishes (success or error).
def _positive_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


JOB_RESULT_RETENTION_SECONDS = _positive_int_env(
    "JOB_RESULT_RETENTION_SECONDS", 3600, minimum=60
)
JOB_CLEANUP_INTERVAL_SECONDS = _positive_int_env(
    "JOB_CLEANUP_INTERVAL_SECONDS", 60, minimum=10
)
