"""Operator protocol and the registry of supported perturbation modes.

In the original code, perturbations were dispatched jointly by *track name*
(ctcf / atac / seq / other) and *ko_mode* inside one if/elif chain. The leaf
logic now lives in ``track_ops`` / ``seq_ops`` / ``deletion``; ``apply_perturbation``
(in this package's ``__init__``) is the faithful top-level dispatcher.
"""
from typing import Protocol

# All ko_modes recognised anywhere in the dispatch. ``increase`` / ``cluster``
# accept a numeric suffix (e.g. ``increase_2.0``, ``cluster_0.05``).
TRACK_MODES = frozenset({
    'zero', 'mean', 'knockout', 'increase', 'cluster',
    'shuffle', 'knockout_shuffle', 'reverse', 'reverse_motif',
})
SEQ_MODES = frozenset({'zero', 'knockout', 'shuffle', 'random', 'reverse', 'reverse_motif'})
DELETION_MODES = frozenset({'del', 'deletion', 'delete'})
SUPPORTED_KO_MODES = TRACK_MODES | SEQ_MODES | DELETION_MODES


def canonical_mode(ko_mode: str) -> str:
    """Strip the numeric suffix from prefix-modes for validation/display.

    ``increase_2.0 -> increase``, ``cluster_0.05 -> cluster``; others unchanged.
    """
    if ko_mode.startswith('increase'):
        return 'increase'
    if ko_mode.startswith('cluster'):
        return 'cluster'
    return ko_mode


class Operator(Protocol):
    """Structural interface for a single perturbation operation.

    Reserved for a future fully object-oriented dispatch; the current engine
    calls the ported functions directly via ``apply_perturbation``.
    """
    name: str

    def apply(self, *args, **kwargs):  # pragma: no cover - interface only
        ...
