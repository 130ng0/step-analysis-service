from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from api import OrcaSliceError, parse_gcode


class GcodeParserTest(TestCase):
    def parse(self, gcode, *, dual=False):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.gcode"
            path.write_text(gcode, encoding="utf-8")
            return parse_gcode(
                str(path),
                filament_metadata={"filament_density_g_cm3": 1, "filament_cost_eur_per_kg": 20},
                support_filament_metadata={"filament_density_g_cm3": 2, "filament_cost_eur_per_kg": 40},
                dual_extruder=dual,
            )

    def test_support_paths_include_arcs_and_leading_decimal_points(self):
        result = self.parse(
            """M83
;TYPE:Outer wall
G1 X1 E1
;TYPE:Support
G1 X2 E.5
G2 X3 Y2 I.5 E+.25 ; E100 is a comment
G1 X4 E-.4
G1 E.4
G1 X5 ; E100 is a comment
;TYPE:Support interface
G3 I1 J0 E.25
G01X6E.5
;TYPE:Inner wall
G1 X7 E1
; filament used [mm] = 4
; filament used [cm3] = 4
; estimated printing time (normal mode) = 2m 30s
"""
        )

        self.assertEqual(result["support_filament_length_mm"], 1.5)
        self.assertEqual(result["model_filament_length_mm"], 2.5)
        self.assertEqual(result["support_filament_weight_g"], 3)
        self.assertEqual(result["material_cost_eur_total"], 0.17)
        self.assertEqual(result["print_time_minutes"], 3)

    def test_absolute_extrusion_tracks_resets_and_mode_changes(self):
        result = self.parse(
            """M82
G92 E10
;TYPE:Outer wall
G1 X2 E12
;TYPE:Support
G1 E11.5
G1 E12
G2 X4 Y2 E12.5
G92 E0
G1 X5 E.25
M83
G1 X6 E.5
M82
G3 X7 Y2 E1
"""
        )

        self.assertEqual(result["support_filament_length_mm"], 1.5)
        self.assertEqual(result["model_filament_length_mm"], 2)
        self.assertEqual(result["filament_length_mm_total"], 3.5)

    def test_support_length_cannot_exceed_rounded_total(self):
        result = self.parse(
            """;TYPE:Support
G1 X1 E1.004
; filament used [mm] = 1.00
"""
        )

        self.assertEqual(result["support_filament_length_mm"], 1)
        self.assertEqual(result["model_filament_length_mm"], 0)

    def test_dual_uses_filament_totals_and_configured_prices(self):
        result = self.parse(
            "; filament used [mm] = 1000, 500\n"
            "; filament used [cm3] = 2.4, 1.2\n"
            "; estimated printing time (normal mode) = 1h 30m\n",
            dual=True,
        )
        self.assertEqual(result["print_time_minutes"], 90)
        self.assertEqual(result["filament_weight_g_total"], 4.8)
        self.assertEqual(result["material_cost_eur_total"], 0.14)
        self.assertEqual([(t["tool"], t["role"]) for t in result["tools"]], [(0, "model"), (1, "support")])

    def test_dual_with_no_support_consumption(self):
        result = self.parse("; filament used [mm] = 1000, 0\n", dual=True)
        self.assertEqual(result["support_filament_weight_g"], 0)
        self.assertEqual(len(result["tools"]), 1)

    def test_dual_rejects_missing_second_total(self):
        with self.assertRaises(OrcaSliceError):
            self.parse("; filament used [mm] = 1000\n", dual=True)
