"""Agent-editable wrapper for the trusted context-ensemble FM implementation."""

from rex.models.context_fm import ContextEnsembleFMPlugin


class ExperimentalContextEnsembleFMPlugin(ContextEnsembleFMPlugin):
    """Constrained research surface; the base implementation is read-only context."""

