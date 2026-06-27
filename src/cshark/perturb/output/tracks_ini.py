"""pyGenomeTracks ``.ini`` builder driven by ``TrackSpec`` objects (scaffold).

The original code built four ``.ini`` files (tmp_tracks / _pred / _true / _diff)
inline across perturb.py lines ~1230-1597, with ``if enformer_active`` /
``if hierarchical_active`` branches leaking model knowledge into rendering.
Here, each secondary predictor instead RETURNS ``TrackSpec`` objects and this
builder just renders whatever specs it is handed.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackSpec:
    """A single track to render in a pyGenomeTracks view."""
    name: str
    path: str                      # bigwig / bed path
    title: str = ''
    color: str = '#666666'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    kind: str = 'bigwig'           # bigwig | bed | arcs


class PyGenomeTracksBuilder:
    """Builds and runs pyGenomeTracks ini files from a list of TrackSpec (scaffold)."""

    def __init__(self, cfg):
        raise NotImplementedError("Port ini construction from perturb.py ~1230-1597.")
