Worker-ready Orca profiles for Nevo3D

What was changed:
- kept folder structure: machine / process / filament
- patched all process and filament profiles to use:
  compatible_printers = ["EL-140V3 v.2.0"]
  compatible_printer_model = ["Generic Marlin Printer"]
- removed empty compatibility condition fields where present
- added supported_nozzle_diameters to process and filament profiles
- normalized machine profile metadata
- added a derived machine profile for 0.6 mm slicing:
  machine/EL-140V3 v.2.0 0.6mm.json

Recommended profile mapping:
- Standard materials (ABS, ABS-CF, ABS-ESD, ASA, PC, PC-CF):
  machine: EL-140V3 v.2.0 0.6mm
  process: Standard 0.6mm v2.0
- Breakaway later:
  machine: EL-140V3 v.2.0 0.6mm
  process: Standard 0.6mm v2.0   (NOT Dualprint, per request)
- Special materials:
  TPU:
    machine: EL-140V3 v.2.0
    process: TPU 0.4mm v2.0
    filament: TPU v.1.1 (REVO) 0.4mm
  PC-FR:
    machine: EL-140V3 v.2.0
    process: Standard 0.4mm v2.0
    filament: PC-FR_Ensinger

Notes:
- These profiles still keep their original Orca 'inherits' chains.
  That means the worker should still resolve/flatten profiles before slicing.
- The zip is intended as a cleaned base to place in the GitHub repo under:
  orca-worker/orca-profiles/