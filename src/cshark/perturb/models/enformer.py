"""Enformer secondary predictor (scaffold).

Wraps the existing ``cshark.inference.utils.enformer_utils`` helpers
(``load_enformer_pretrained`` / ``load_enformer_from_checkpoint`` /
``enformer_seq_knockout`` / ``write_tmp_enformer_*_bigwig``). Ports the
enformer block of the original ``single_deletion`` (perturb.py lines 925-963):
on the mutated sequence, predict a 1D delta, rewrite the perturbed input
track(s), and return TrackSpecs for the predicted-KO and delta bigwigs.
"""
from cshark.perturb.models.base import SecondaryPredictor


class EnformerSecondaryPredictor:  # implements SecondaryPredictor
    def __init__(self, cfg):
        raise NotImplementedError("Port enformer setup/load from perturb.py lines 925-943.")

    def prepare(self, wt):
        pass

    def apply(self, wt, ko):
        raise NotImplementedError("Port enformer_seq_knockout + bigwig writes (perturb.py 936-962).")
