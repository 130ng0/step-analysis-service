from typing import List, Optional
from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    success: bool = True
    filename: Optional[str] = None
    source_format: str = Field(...)

    unit: str = "mm"

    volume_mm3: float
    volume_cm3: float
    surface_mm2: float

    bbox_x_mm: float
    bbox_y_mm: float
    bbox_z_mm: float

    solid_count: int = 1
    is_closed_solid: bool = True
    warnings: List[str] = []

    machine_hour_rate_eur: Optional[float] = None
    volumetric_flow_mm3_s: Optional[float] = None

    estimated_print_seconds: Optional[float] = None
    estimated_print_hours: Optional[float] = None
    estimated_machine_cost_eur: Optional[float] = None

    support_angle_deg: Optional[float] = None
    support_density_factor: Optional[float] = None
    support_required: Optional[bool] = None
    critical_overhang_area_mm2: Optional[float] = None
    average_support_height_mm: Optional[float] = None
    support_volume_mm3: Optional[float] = None
    support_volume_cm3: Optional[float] = None

    estimated_total_print_seconds: Optional[float] = None
    estimated_total_print_hours: Optional[float] = None
    estimated_total_machine_cost_eur: Optional[float] = None

    part_material_name: Optional[str] = None
    part_material_price_per_kg_eur: Optional[float] = None
    part_material_density_g_cm3: Optional[float] = None

    support_material_type: Optional[str] = None
    support_material_name: Optional[str] = None
    support_material_price_per_kg_eur: Optional[float] = None
    support_material_density_g_cm3: Optional[float] = None

    part_weight_g: Optional[float] = None
    support_weight_g: Optional[float] = None

    part_material_cost_eur: Optional[float] = None
    support_material_cost_eur: Optional[float] = None
    total_material_cost_eur: Optional[float] = None

    subtotal_cost_eur: Optional[float] = None
    margin_factor: Optional[float] = None
    total_price_eur: Optional[float] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    filename: Optional[str] = None