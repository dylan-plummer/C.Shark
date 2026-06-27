"""Hierarchical RAD21 secondary predictor (scaffold).

Wraps ``cshark.inference.utils.hierarchical_utils``
(``load_hierarchical_rad21_predictor`` / ``hierarchical_rad21_update`` /
``write_tmp_hierarchical_*_bigwig``). Ports the hierarchical blocks that the
original code duplicated across BOTH scope paths:

- full-chrom: perturb.py lines 326-362 (setup), 374-389 (predict + insert
  missing rad21), 410-511 (delta collection).
- single-locus: perturb.py lines 706-761 (setup/insert), 975-1019 (delta).

``prepare`` predicts RAD21 and inserts it into the inputs when the main model
needs a rad21 channel that wasn't provided; ``apply`` computes the WT-vs-KO
RAD21 delta, rewrites the experimental rad21 track, and returns TrackSpecs.
"""
from cshark.perturb.models.base import SecondaryPredictor


class HierarchicalRad21Predictor:  # implements SecondaryPredictor
    def __init__(self, cfg):
        raise NotImplementedError("Port load_hierarchical_rad21_predictor setup (perturb.py 334-356 / 710-761).")

    def prepare(self, wt):
        raise NotImplementedError("Predict + insert missing rad21 input track (perturb.py 374-389 / 716-761).")

    def apply(self, wt, ko):
        raise NotImplementedError("Port hierarchical_rad21_update + bigwig writes (perturb.py 426-435 / 996-1019).")
