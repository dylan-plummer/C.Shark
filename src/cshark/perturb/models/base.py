"""Model-agnostic seams: ``PerturbationModel`` and ``SecondaryPredictor``.

``CSharkModel`` wraps the default 8-track model. It calls ``model_utils.load_default``
exactly ONCE in ``__init__`` and caches the model -- fixing the original bug
where ``infer.prediction`` reloaded and re-detected the checkpoint on *every*
sliding window (perturb.py lines 397/439 inside the full-chrom loop, and again
per call in single_deletion). The forward + post-processing logic is a faithful
port of the body of ``inference_utils.prediction`` (lines 432-483); the shared
helpers ``load_default`` / ``preprocess_default`` are reused as-is (imported),
so ``inference_utils`` itself is untouched.
"""
from typing import List, Optional, Protocol

import numpy as np
import torch

from cshark.inference.utils.inference_utils import preprocess_default
from cshark.inference.utils.model_utils import load_default, get_1d_track_names
from cshark.perturb.config import WINDOW
from cshark.perturb.context import RegionInputs

# 1D heads the dict-API model is asked to predict (verbatim from prediction()).
_PREDICT_TRACKS = ['ctcf', 'atac', 'rad21', 'h3k27ac', 'h3k4me3', 'h3k9me3',
                   'h3k36me3', 'h3k27me3', 'myc', 'nanog', 'yy1', 'polr2a', 'rnaseq', 'hic']


class PerturbationModel(Protocol):
    """Anything that maps preprocessed region inputs to predictions."""
    track_names_1d: List[str]

    def predict(self, inputs: RegionInputs) -> dict:   # {'hic': np.ndarray, '1d': np.ndarray | None}
        ...


class SecondaryPredictor(Protocol):
    """A model that mutates input tracks before the main model runs.

    ``prepare`` runs once on the WT inputs (e.g. hierarchical fills a missing
    rad21 input track by predicting it). ``apply`` rewrites the perturbed (KO)
    inputs from a predicted delta, writes any tmp bigwigs it wants visualised,
    and returns ``TrackSpec`` objects for the renderer.
    """
    def prepare(self, wt: RegionInputs) -> None: ...

    def apply(self, wt: RegionInputs, ko: RegionInputs) -> list:  # -> list[TrackSpec]
        ...


class CSharkModel:
    """Adapter around the default 8-track C.Shark model (loaded once)."""

    def __init__(self, cfg, num_genomic_features, diploid=False):
        self.model = load_default(
            cfg.model_path, record_attn=False,
            num_genomic_features=num_genomic_features,
            mat_size=cfg.mat_size, diploid=diploid,
            mid_hidden=cfg.mid_hidden,
            target_1d_length=int(WINDOW / cfg.resolution_1d),
            seq_filter_size=cfg.seq_filter_size, recon_1d=cfg.recon_1d,
        )
        self.bigwig_log = cfg.bigwig_log_transform     # controls 1D expm1
        self.undo_log = cfg.hic_log_transform          # controls Hi-C expm1
        try:
            self.track_names_1d = get_1d_track_names(cfg.model_path) or []
        except Exception:
            self.track_names_1d = []

    def predict(self, inputs: RegionInputs) -> dict:
        # other_feat_names mirror the original ``input_track_names[2:]`` slice.
        other_feat_names = inputs.track_names[2:] if inputs.track_names else None
        return self._predict_arrays(inputs.seq, inputs.ctcf, inputs.atac,
                                    inputs.others, other_feat_names)

    def _predict_arrays(self, seq_region, ctcf_region, atac_region, other_regions, other_feat_names):
        """Faithful port of the non-attn body of ``inference_utils.prediction``."""
        if other_regions is None:
            inputs = preprocess_default(seq_region, ctcf_region, atac_region)
        else:
            inputs = preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
        try:
            output = self.model(inputs)
        except TypeError:
            inputs = {
                'seq': inputs[..., :5],
                'ctcf': inputs[..., 5:6],
                'atac': inputs[..., 6:7],
            }
            if other_feat_names is not None and other_regions is not None:
                for i, other_feat in enumerate(other_feat_names):
                    inputs[other_feat] = torch.tensor(other_regions[i]).unsqueeze(0).unsqueeze(2).to(inputs['seq'].device)
            output = self.model(inputs, predict_tracks=_PREDICT_TRACKS)

        if isinstance(output, dict):
            pred = output['hic']
            pred_1d = output['1d']
            if pred_1d is not None:
                if isinstance(pred_1d, dict):
                    tmp_1d = [v.detach().cpu().numpy() for v in pred_1d.values()]
                    pred_1d = np.concatenate(tmp_1d, axis=0).transpose()
                else:
                    pred_1d = pred_1d[0].detach().cpu().numpy()
                if self.bigwig_log:
                    pred_1d = np.expm1(pred_1d)
                    pred_1d = np.clip(pred_1d, 0, None)
            else:
                pred_1d = None
        else:
            pred = output
            pred_1d = None
        pred = pred[0].detach().cpu().numpy()
        pred = (pred + pred.T) * 0.5                   # symmetrize
        if self.undo_log:
            pred = np.expm1(pred)
        return {'hic': pred, '1d': pred_1d}
