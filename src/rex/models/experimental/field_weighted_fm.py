"""Agent-editable wrapper for the trusted field-weighted FM implementation."""

from rex.models.field_weighted_fm import FieldWeightedFMPlugin


class ExperimentalFieldWeightedFMPlugin(FieldWeightedFMPlugin):
    """Constrained research surface for field-pair weighting experiments."""
