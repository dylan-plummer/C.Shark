"""Region input container passed through the perturbation pipeline.

``RegionInputs`` bundles the model inputs for one genomic window (one-hot
sequence + the 1D feature tracks) together with their names, so that operators,
secondary predictors, and the main model can address tracks by name instead of
juggling separate ``ctcf_region`` / ``atac_region`` / ``other_regions`` variables
and ad-hoc channel offsets (as the original code did).

NOTE: field set will be finalised while porting ``scopes/`` against the old
``single_deletion`` / full-chrom data flow; the core fields below match what
``load_region`` returns today.
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np


@dataclass
class RegionInputs:
    seq: np.ndarray                                  # (L, 5) one-hot, or (L, 10) diploid
    ctcf: Optional[np.ndarray] = None
    atac: Optional[np.ndarray] = None
    others: Optional[List[np.ndarray]] = None        # extra 1D tracks (e.g. rad21, h3k27ac)
    track_names: List[str] = field(default_factory=list)   # input track order, e.g. ['ctcf','atac','rad21']
    track_paths: List[str] = field(default_factory=list)

    def copy(self) -> "RegionInputs":
        """Deep-ish copy: arrays are copied so KO does not mutate the WT inputs."""
        return replace(
            self,
            seq=None if self.seq is None else self.seq.copy(),
            ctcf=None if self.ctcf is None else self.ctcf.copy(),
            atac=None if self.atac is None else self.atac.copy(),
            others=None if self.others is None else [a.copy() for a in self.others],
            track_names=list(self.track_names),
            track_paths=list(self.track_paths),
        )

    def channel_index(self, name: str) -> int:
        """Index of track ``name`` within ``track_names`` (raises if absent)."""
        return self.track_names.index(name)

    def other_index(self, name: str) -> int:
        """Index of track ``name`` within ``others`` (i.e. excluding ctcf/atac)."""
        base = [t for t in self.track_names if t not in ('ctcf', 'atac')]
        return base.index(name)
