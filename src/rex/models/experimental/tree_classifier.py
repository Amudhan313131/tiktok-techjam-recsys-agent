"""Agent-editable adapter for the trusted pointwise LightGBM candidate."""

from rex.models.tree_classifier import TreeClassifierPlugin


class ExperimentalTreeClassifierPlugin(TreeClassifierPlugin):
    """Constrained research surface for regularized pointwise tree experiments."""
