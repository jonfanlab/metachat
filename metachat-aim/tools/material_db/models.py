from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from uuid import UUID

class MaterialCategory(Enum):
    MAIN = "main"
    GLASS = "glass"
    ORGANIC = "organic"
    OTHER = "other"

class MaterialType(Enum):
    BULK = "bulk"
    FILM = "film"
    CRYSTAL = "crystal"
    GLASS = "glass"
    LIQUID = "liquid"

@dataclass
class MaterialData:
    id: Optional[UUID]
    name: str
    category: MaterialCategory
    type: MaterialType
    wavelength_range: tuple[float, float]
    wavelength_unit: str
    data_type: str  # 'tabulated n', 'tabulated k', 'tabulated nk', 'formula'
    reference: Dict[str, Any]
    specs: Dict[str, Any]
    file_path: str
    data: Dict[str, Any]
    measurements: List[tuple[float, float, Optional[float]]]  # (wavelength, n, k) tuples