"""Feature bundle with explicit point-in-time provenance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.data.views import ENGINEERED_PREFIX, FeatureView, load_feature_view


@dataclass(frozen=True)
class FeatureBundle:
    arrays: dict[str, np.ndarray]
    provenance: dict[str, dict[str, object]]

    def validate(self, rows: int) -> None:
        for name, values in self.arrays.items():
            if len(values) != rows:
                raise ValueError(f"feature {name} has {len(values)} rows; expected {rows}")
            if name not in self.provenance:
                raise ValueError(f"feature {name} is missing provenance")
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise ValueError(f"feature {name} contains non-finite values")

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        array_path = output / "features.npz"
        provenance_path = output / "provenance.json"
        np.savez_compressed(array_path, **self.arrays)
        provenance_path.write_text(
            json.dumps(self.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return array_path, provenance_path


def attach_feature_bundles(
    base: FeatureView | str | Path,
    bundles: list[FeatureBundle],
    destination: str | Path,
) -> Path:
    """Materialize hash-covered engineered columns into a sanitized model view."""
    view = base if isinstance(base, FeatureView) else load_feature_view(base)
    arrays = dict(view.arrays)
    provenance: dict[str, object] = {}
    for bundle in bundles:
        bundle.validate(view.rows)
        for name, values in bundle.arrays.items():
            column = name if name.startswith(ENGINEERED_PREFIX) else ENGINEERED_PREFIX + name
            if column in arrays:
                raise ValueError(f"duplicate engineered feature: {column}")
            values = np.asarray(values)
            if values.ndim != 1 or values.dtype.kind not in "biuf":
                raise ValueError(
                    f"invalid engineered feature {column}: shape={values.shape}, dtype={values.dtype}"
                )
            arrays[column] = values
            provenance[column] = bundle.provenance.get(name, {})
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_feature_view(path)
    return path
