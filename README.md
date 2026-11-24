# 3D Model Analysis Service

## Orca worker 2.5.2

* The worker uses the official OrcaSlicer 2.5.2 AppImage for Ubuntu 24.04 on Linux x86-64. The Docker build verifies its SHA-256 checksum.
* The G-code parser accounts for linear and arc extrusion, decimal values such as `E.59691`, relative and absolute extrusion, and `G92` resets. Model and support material are calculated separately for pricing.
* The [initial orientation test](#orientation-test) used the explicit `--orient 1` option. The current worker uses the fast orientation estimate described below.
* Local Docker verification passed for the `/slice` endpoint with ABS and PC-FR and for the [G-code parser regression tests](orca-worker/test_gcode_parser.py).

See [Orca worker deployment](#orca-worker-deployment) for the Portainer update and rollback steps.

## 2.14.4 / Orca Worker API 2.3.4 - Shell settings from the process profile

If a client omits `perimeter_count`, `top_layers` or `bottom_layers`, Orca uses the corresponding setting from the resolved process profile. Explicit overrides, including zero layers, are accepted by both analysis endpoints. Job results report the effective values in `applied_slicer_settings`. Rebuild both service containers before updating Odoo to 17.0.1.8.30.

## 2.14.3 / Orca Worker API 2.3.3 - Nozzle selection and dual extrusion

Both slice API paths accept optional `nozzle_diameter_mm` (`0.4` or `0.6`). ABS, ABS-CF, ABS-ESD, ASA, PC and PC-CF have both sizes. PC-FR and TPU use 0.4 mm. Omitted sizes use the previous material defaults.

Soluble support and HIPS use the supplied EL-140V3 Dualprint profile family, with separate model and support filaments. The model uses tool 0, while both support body and interface use tool 1. The worker strips custom G-code macros, printer connection settings and post-processing hooks from those profiles. The prime tower is disabled. Breakaway and `none` use a single extruder. `none` disables support generation.

The parser uses Orca's per-filament G-code footer totals for Dualprint, applying the density and price supplied by Odoo. Legacy output fields remain available and `tools` reports the tool index and costs per filament. The result also records the chosen nozzle and configured extruder count. Rebuild both containers and update Odoo to 17.0.1.8.28 to enable website nozzle selection.

## 2.14.2 / Orca Worker API 2.3.2 - Fast orientation estimate

With supports enabled, the worker reads binary STL bounds and samples up to 32,768 triangles. It estimates the support needed in the input orientation and after a 180-degree rotation around X. It flips only when the estimated support score drops by more than 30 percent and the rotated model has a measurable bed contact area. ASCII STL and geometry without a clear preference use the input orientation.

The worker runs Orca once. The chosen orientation is recorded in `applied_slicer_settings.orientation` as `original` or `flipped_x_180`. This setting describes the slice used for pricing. The preview shows the uploaded orientation. Printable contact area and support requirements are still estimates until Orca runs. Rebuild both the `orca-worker` and `step-analysis-service` images to enable orientation selection and to retain the chosen orientation in queued job results.

## Orca Worker API 2.3.1 - Soluble support estimates

* Soluble support uses zero top and bottom Z gaps. Breakaway uses the process profile's Z gaps.
* For support volume measured from the G-code, the worker uses the supplied support price and density before falling back to profile values. Both support body and interface are charged as the selected support material.
* The estimate slices with one extruder. Tool-change and purge time and waste are outside the estimate.
* Rebuild the `orca-worker` image before updating the Odoo website module to 17.0.1.8.26. The job API container uses the existing request contract.

## 2.14.1 - Queue time estimates

Queued jobs include an approximate remaining time in `eta_seconds`. The estimate combines the current phases of running jobs, the number of live workers and a 90-second allowance for each waiting or not-yet-estimated job. Queue positions and job claiming use the same FIFO ordering, with insertion order breaking timestamp ties. The Orca worker image is independent of this API update.

## 2.14.0 - Brighter balanced preview

* Preview surfaces are one step lighter than 2.13.0 for clearer geometry on thermal labels while preserving the CAD shading on screen.
* Black feature and silhouette edges, white background, render resolution, cropping, and portrait storage are unchanged.

## 2.13.0 - Screen + thermal-label preview contrast

* Preview surfaces use a lighter neutral CAD gray so shading remains attractive on screen while black feature/silhouette edges retain strong contrast on thermal labels.
* Darkest surface tones are kept clearly separated from the black edge pass; background remains white.
* Portrait storage/cropping behavior from 2.12.0 is unchanged.

## 2.12.0 - Portrait preview storage

- Final preview PNGs are whitespace-cropped first.
- If the cropped preview is wider than high, it is rotated 90 degrees clockwise before storage.
- Rotation is lossless (Pillow transpose): no scaling or resampling; width/height are only swapped.
- Portrait and square previews remain unchanged.
- Internal VTK render resolution remains 700 x 560 px before cropping.


## Overview

The 3D Model Analysis Service accepts uploaded STEP or STL files, converts them when necessary, slices them using an Orca worker, calculates material and machine costs, generates a preview image, and exposes the result through an asynchronous job API.

The service is designed for integration with Odoo and similar systems that need reliable, bounded parallel processing of potentially expensive slicing jobs.

---

## Core Capabilities

* Upload STEP and STL files
* Automatic STEP to STL conversion
* Configurable parallel job queue processing (1-10 workers; default 5)
* Orca slicer integration through a dedicated worker
* Preview image generation from STL
* Material, support, and machine cost calculation
* Polling-based job API for asynchronous integrations
* Immediate cleanup of uploaded/derived job files on success and error
* Idempotent terminal-result handoff with bounded SQLite retention

---

## V3.5 isolated headless preview renderer

* VTK/OpenGL preview rendering now runs in a dedicated child process instead of inside the FastAPI process.
* Production Docker runs that child under `xvfb-run`, providing a deterministic headless X/OpenGL context.
* A native VTK/OpenGL abort or segfault can no longer terminate the API service on port 5050.
* Preview rendering remains serialized while STEP conversion and Orca slicing can use the configured parallel worker pool.
* `PREVIEW_RENDER_TIMEOUT_SECONDS` (default `120`) bounds hung renderer processes.
* The Docker image now installs Xvfb, xauth and Mesa DRI explicitly.

## V3.4 topology-correct CAD edge rendering

* STEP-to-STL mesh vertices are welded with `vtkCleanPolyData` before CAD edge extraction. This prevents duplicated STL triangle vertices from being mistaken for boundaries and preserves real geometric transitions.
* Surface normals remain a separate shading pipeline; edge extraction runs on the cleaned original topology rather than the split-normal mesh.
* Feature edges use an 18 degree dihedral threshold to retain relevant chamfers, recesses, holes and steps without exposing ordinary STL triangulation.
* Feature edges and silhouettes are rendered as thin 3D tubes. They therefore participate in the normal depth buffer and cannot bleed through opaque front surfaces like line primitives can.
* Tube radius scales with the model bounding-box diagonal so edge weight remains visually stable across small and large parts.
* Validated against the supplied multi-part STEP regression set, including Raspberry cases, brackets, clamps, transport locks and sled components.

## V3.3 depth-correct CAD preview rendering

* Fully opaque dark-charcoal CAD surfaces with directional lighting.
* VTK off-screen rendering uses a real depth buffer, so hidden/back-side engraving, text and geometry cannot bleed through front faces.
* Depth-tested feature edges highlight holes, slots, ribs, recesses and sharp corners without showing hidden wireframe geometry.
* A separate silhouette pass guarantees a strong, complete outside contour, including shallow-angle outline segments.
* Orthographic CAD projection keeps compact Odoo thumbnails geometrically readable.
* Preview rendering remains serialized for graphics-context safety; analysis/slicing workers remain parallel.

## V3.1 reliability / cleanup

The service now separates **working files** from **terminal result metadata**:

* `input.bin`, converted artefacts, preview helper files, and other per-job files are deleted **immediately when a job finishes**, both for `done` and `error`.
* The small SQLite terminal row is kept only long enough for Odoo to fetch the result idempotently. Odoo acknowledges the result with `DELETE /analyze-model/jobs/{job_id}` after its own database transaction has committed.
* As a safety net, stale terminal rows are removed automatically after `JOB_RESULT_RETENTION_SECONDS` (default `3600`).
* Job state is stored on the persistent Docker volume `step-analysis-job-state`, so queued jobs survive a container restart. Jobs that were `processing` are requeued on startup if their input is still present.
* Preview rendering uses VTK off-screen rendering with a real depth buffer. Hidden/back-side edges are occluded correctly, while a separate silhouette pass keeps the complete outer contour visible. Rendering stays serialized; slicing remains parallel.

Environment variables:

```text
ANALYSIS_MAX_WORKERS=5             # allowed 1..10
JOB_RESULT_RETENTION_SECONDS=3600  # terminal DB safety retention
JOB_CLEANUP_INTERVAL_SECONDS=60
PREVIEW_RENDER_TIMEOUT_SECONDS=120
```

## Production Architecture

### Components

#### 1. Public API Service

This FastAPI application:

* accepts incoming jobs
* stores job payloads and uploaded files
* exposes status and result endpoints
* starts the internal job worker loop on startup

#### 2. Internal Job Worker

A background worker loop:

* atomically claims the next queued job
* converts STEP to STL if needed
* renders a preview image
* calls the Orca worker for slicing
* stores the final result or error state

#### 3. Orca Worker

A dedicated slicing service that:

* receives STL files and slicing parameters
* resolves machine, process, and filament profiles
* slices with OrcaSlicer
* parses G-code for time, material usage, and cost details

#### 4. Job Store

SQLite-backed job persistence for:

* queued jobs
* running jobs
* completed jobs
* failed jobs
* cleanup after consumption

### V3.3 visibility fixes

* Fully opaque CAD surfaces; no alpha-based face rendering.
* VTK depth-buffer rendering prevents back-side engraving, text and hidden geometry from bleeding through front faces.
* Visible feature edges are depth-tested against the solid model.
* A dedicated silhouette pass draws complete outside contours, including shallow-angle outline segments.
* Orthographic CAD camera and dark charcoal shading retained for compact Odoo thumbnails.

---

## High-Level Request Flow

1. Client uploads STEP or STL file to `POST /analyze-model/jobs`
2. Service stores file and metadata, returns a `job_id`
3. One of the configured background workers atomically claims the next queued job
4. STEP files are converted to STL
5. Preview image is rendered from STL
6. STL is sent to Orca worker for slicing
7. Result is stored under the job
8. Client polls `GET /analyze-model/jobs/{job_id}`
9. Once complete, client retrieves the result
10. Client calls `DELETE /analyze-model/jobs/{job_id}` to clean up

---


## V3 Parallel Analysis Workers

The analysis service can process several jobs concurrently. Configure the pool with:

```bash
ANALYSIS_MAX_WORKERS=5
```

Valid effective values are **1 to 10**. The default is **5** and values above 10 are clamped to 10 to avoid accidentally launching an unbounded number of OrcaSlicer processes.

Job claiming is atomic at SQLite level using `BEGIN IMMEDIATE` plus a guarded status update. This prevents two workers (including workers from separate service processes sharing the same database) from processing the same queued job.

The Orca worker moves the blocking OrcaSlicer subprocess into a thread so concurrent HTTP slice requests can actually execute in parallel. Keep the worker count appropriate for available CPU and RAM.

`GET /health` reports `analysis_workers_configured` and `analysis_workers_alive`.

---

## API Endpoints

## Health Check

```http
GET /health
```

### Example Response

```json
{
  "status": "ok",
  "orca_worker": "ok"
}
```

---

## Create Analysis Job

```http
POST /analyze-model/jobs
```

### Content Type

`multipart/form-data`

### Form Fields

#### File

* `file`: STEP or STL file

#### Material Mapping

* `material_profile`
* `support_material_type`
* `material_display_name`
* `support_material_display_name`

#### Slicer Settings

* `infill_percent`
* `perimeter_count`
* `top_layers`
* `bottom_layers`

#### Pricing Data

* `machine_hour_rate_eur`
* `margin_factor`
* `material_density_g_cm3`
* `material_price_eur_per_kg`
* `support_density_g_cm3`
* `support_price_eur_per_kg`

### Example Response

```json
{
  "success": true,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "status": "queued"
}
```

---

## Get Job Status

```http
GET /analyze-model/jobs/{job_id}
```

### Queued Response

```json
{
  "success": true,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "status": "queued",
  "queue_position": 2
}
```

### Processing Response

```json
{
  "success": true,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "status": "processing"
}
```

### Done Response

```json
{
  "success": true,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "status": "done",
  "result": {
    "success": true,
    "filename": "part.step",
    "method": "slice",
    "material_profile": "abs",
    "support_material_type": "breakaway",
    "unit": "mm",
    "machine_hour_rate_eur": 8.0,
    "margin_factor": 1.0,
    "print_time_minutes": 54,
    "print_time_hours": 0.9,
    "filament_length_mm_total": 7348.29,
    "filament_volume_cm3_total": 17.67,
    "filament_weight_g_total": 18.377,
    "material_cost_eur_total": 0.51,
    "machine_cost_eur": 7.2,
    "subtotal_cost_eur": 7.71,
    "total_price_eur": 7.71,
    "preview_png_base64": "..."
  }
}
```

### Error Response

```json
{
  "success": false,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "status": "error",
  "error": "SLICE_FAILED",
  "details": "Orca worker failed: ..."
}
```

---

## Delete Job

```http
DELETE /analyze-model/jobs/{job_id}
```

### Example Response

```json
{
  "success": true,
  "job_id": "4c4378fd-bcb7-4c0b-a8d7-7cc4d7caa4fb",
  "deleted": true
}
```

This endpoint should be called after the client has persisted the result.

---

## Request and Result Semantics

### Material Pricing Source of Truth

The service expects pricing and density values from Odoo or another upstream system.

This means:

* slicing profiles are used for slicing behavior
* pricing and density are controlled externally
* the service does not need to be the master source for material economics

### Support Material Modes

Supported support material types:

* `none`
* `breakaway`
* `hips`
* `soluble`

Typical meaning:

* `none`: no support pricing
* `breakaway`: same pricing and density as part material unless overridden
* `hips`: support-specific pricing and density
* `soluble`: support-specific pricing and density

---

## Preview Rendering

Preview images are generated after STL conversion and before slicing result assembly.

### Output Field

```json
"preview_png_base64": "..."
```

### Notes

* preview is a static PNG
* preview is rendered from STL, not STEP directly
* preview rendering failure should not block slicing results

---

## Directory Layout

Recommended service structure:

```text
app/
  main.py
  config.py
  schemas.py
  security.py
  job_store.py
  worker_loop.py
  services/
    slice_input_converter.py
    model_analysis.py
    orca_client.py
```

Runtime storage in Docker Compose:

```text
/data/
  jobs.db
  files/
    <job_id>/
      input.bin
      input_name.txt
      preview_base64.txt
```

The per-job directory exists only while the job is queued/processing and is deleted immediately on terminal success or error.

---

## Environment Variables

### API Service

```env
ORCA_WORKER_URL=http://orca-worker:8090
ORCA_WORKER_TIMEOUT=1800
ANALYSIS_MAX_WORKERS=5
JOB_DB_PATH=/data/jobs.db
JOB_FILES_DIR=/data/files
JOB_RESULT_RETENTION_SECONDS=3600
JOB_CLEANUP_INTERVAL_SECONDS=60
PREVIEW_RENDER_TIMEOUT_SECONDS=120
```

### Orca Worker

Per-run Orca temporary data is always deleted in a `finally` block on both success and failure.

---

## Installation

## Python Version

Recommended:

* Python 3.11 or 3.12

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start API Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5050
```

---

## Recommended `requirements.txt`

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.22
requests==2.32.5
cadquery==2.4.0
trimesh==4.4.9
numpy==1.26.4
matplotlib==3.9.0
pydantic~=2.12.5
```

The CAD preview renderer uses VTK off-screen rendering. `vtk>=9.2,<10` is declared explicitly even though CadQuery installations commonly bring VTK transitively.

---

## Docker Notes

### Orca worker deployment

Compose builds the Orca worker for Linux x86-64 from `orca-worker/Dockerfile`. The Dockerfile downloads and verifies the official OrcaSlicer 2.5.2 AppImage for Ubuntu 24.04. For the first deployment of version reporting, rebuild both services from the updated checkout with the existing stack environment:

```sh
docker compose up -d --build step-analysis-service orca-worker
curl -fsS http://localhost:5050/health
```

### Portainer Git stack

1. Push the updated worker and Compose files to the branch configured for the stack.
2. In Portainer, select **Pull and redeploy** to fetch the Git revision. **Re-pull image** is optional for this rollout. With `pull_policy: build`, Compose builds the worker from its Dockerfile even if the previous image is cached.
3. Request `http://<docker-host>:5050/health` and check for `"orca_worker": "ok"` and `"orca_slicer_version": "2.5.2"` to verify the update.

For subsequent updates, edit the stack environment variables in Portainer and redeploy with a build:

- `ORCA_SLICER_VERSION` selects the stable release, for example `2.5.2` or `2.5.1`. The default is `2.5.2`.
- `ORCA_SLICER_SHA256` supplies the official SHA-256 digest for that release's Ubuntu 24.04 x86-64 AppImage. For `2.5.2` and `2.5.1`, leave it empty to use the pinned checksum. For other versions, enter the 64 hexadecimal characters without the `sha256:` prefix.

The build verifies the AppImage before extracting it. Select `2.5.1` and clear the checksum override to rebuild with the previous slicer. For a quick rollback, use the retained prebuilt image. Complete running analyses before replacing the worker.

If Portainer reports `pull access denied for nevo-orca-worker`, check the configured Git branch and Compose path. The current [Compose file](docker-compose.yml) uses a `build:` block and `pull_policy: build` for the worker. A stack definition with `image: nevo-orca-worker:2.5.2` can make Portainer try Docker Hub for that tag.

Use the existing Compose project name and stack environment when updating a Portainer-managed stack from the command line.

### Prebuilt image and rollback

For deployment from the prebuilt local export, transfer `orca-worker-2.5.2-linux-amd64.tar` to the Docker host and run `docker load -i orca-worker-2.5.2-linux-amd64.tar`. Tag `nevo-orca-worker:2.5.2` with the worker image name reported by `docker compose config --images` for the existing project, then run `docker compose up -d --no-build --pull never --no-deps orca-worker`. The image archive is excluded from regular Git commits. It can be uploaded as a GitHub Release asset if needed.

To return to 2.5.1, load the retained `orca-worker-2.5.1-linux-amd64.tar` and follow the prebuilt deployment procedure with `nevo-orca-worker:2.5.1`. To return to 2.3.2 by rebuilding, restore [`archive/orca-2.3.2/Dockerfile`](archive/orca-2.3.2/Dockerfile) to `orca-worker/Dockerfile` before rebuilding the worker. To deploy the archived image, load it, tag `nevo-orca-worker:2.3.2` with the existing project's worker image name, and use the prebuilt deployment command above.

### Orientation test

In a local comparison on the SC6-1X90 adapter with PC-FR, a 0.4 mm nozzle, 20% infill, and breakaway supports, OrcaSlicer 2.5.1 estimated 173.48 g of supports and 21 h 32 min in the original orientation. Explicit `--orient 1` selected the same orientation as a manual 180° rotation about X and reduced the estimates to 24.97 g and 14 h 38 min. OrcaSlicer 2.3.2 also selected that orientation when explicitly enabled. The estimates apply to this model and profile.

### System Dependencies

For preview rendering and CAD stack support, containers often need additional OS packages.

Typical requirements:

```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

### Docker Compose ports

The [API Dockerfile](Dockerfile) listens on container port `8080`. The [Compose stack](docker-compose.yml) publishes it on host port `5050`. The API reaches the worker over the internal Compose network at `http://orca-worker:8090`.

---

## Deployment Guidance

## Single Instance Requirement

The current SQLite-based queue is intended for a single API instance.

Do not horizontally scale the API service without replacing the job store and queue coordination mechanism.

### Good for

* VPS deployment
* single Docker host
* small internal production setup

### Not yet ideal for

* multiple replicas
* Kubernetes horizontal scaling
* distributed workers without shared coordination

## If scaling later

Move to one of:

* Postgres-backed job coordination
* Redis-backed queue
* dedicated task framework such as Celery or RQ

---

## Reliability Notes

### Why the worker loop uses background execution

The job worker must not block FastAPI request handling. Long-running conversion, rendering, and slicing tasks should run outside the main event loop.

### Cleanup Strategy

Recommended cleanup policy:

* client deletes job after persisting the result
* service deletes uploaded files and stored result
* optional TTL cleanup for abandoned jobs

### Suggested TTL

* delete unconsumed done/error jobs after 24 hours

---

## Odoo Integration

## Recommended Odoo Flow

1. User uploads file in Odoo
2. Odoo calls `POST /analyze-model/jobs`
3. Odoo stores `service_job_id`
4. User or cron polls `GET /analyze-model/jobs/{job_id}`
5. When `done`, Odoo stores:

   * print time
   * filament usage
   * cost data
   * preview image
6. Odoo calls `DELETE /analyze-model/jobs/{job_id}`
7. Odoo clears remote job reference

## Odoo Fields Typically Stored

* `service_job_id`
* `service_job_status`
* `service_job_queue_position`
* `service_job_last_poll`
* `preview_image`
* pricing result fields
* tool breakdown JSON

---

## Example cURL Requests

## Create Job

```bash
curl -X POST \
  'http://localhost:5050/analyze-model/jobs' \
  -H 'accept: application/json' \
  -H 'x-api-key: YOUR_API_KEY' \
  -H 'Content-Type: multipart/form-data' \
  -F 'material_profile=abs' \
  -F 'support_material_type=breakaway' \
  -F 'infill_percent=20' \
  -F 'perimeter_count=5' \
  -F 'top_layers=5' \
  -F 'bottom_layers=5' \
  -F 'machine_hour_rate_eur=8' \
  -F 'margin_factor=1' \
  -F 'material_density_g_cm3=1.04' \
  -F 'material_price_eur_per_kg=28' \
  -F 'support_density_g_cm3=1.04' \
  -F 'support_price_eur_per_kg=28' \
  -F 'material_display_name=ABS Pro' \
  -F 'support_material_display_name=Breakaway' \
  -F 'file=@part.step'
```

## Poll Job

```bash
curl -X GET \
  'http://localhost:5050/analyze-model/jobs/JOB_ID' \
  -H 'accept: application/json' \
  -H 'x-api-key: YOUR_API_KEY'
```

## Delete Job

```bash
curl -X DELETE \
  'http://localhost:5050/analyze-model/jobs/JOB_ID' \
  -H 'accept: application/json' \
  -H 'x-api-key: YOUR_API_KEY'
```

---

## Error Handling

Common error codes:

* `UNSUPPORTED_FILE_FORMAT`
* `FILE_TOO_LARGE`
* `SLICE_INPUT_CONVERSION_FAILED`
* `SLICE_FAILED`
* `ORCA_WORKER_UNREACHABLE`
* `JOB_NOT_FOUND`
* `INTERNAL_SERVER_ERROR`

Recommended client behavior:

* show user-friendly message for validation errors
* retry polling for `queued` and `processing`
* stop polling on `done` or `error`
* persist `result` immediately before deleting the job

---

## Security

* protect endpoints with API key validation
* avoid exposing raw internal file paths
* delete processed files after consumption
* consider upload size limits and rate limiting in production

---

## Observability

Recommended production additions:

* structured JSON logging
* request IDs in logs
* job lifecycle logs
* metrics for queue length and processing times
* alerting for repeated slice failures

---

## Future Improvements

* automatic stale job cleanup
* retry policy for transient Orca worker failures
* Postgres-backed queue state
* WebSocket or server-sent events for live status updates
* multiple worker priorities
* preview rendering enhancements
* job cancellation support

---

## License / Ownership

Internal project for Nevo3D GmbH.
