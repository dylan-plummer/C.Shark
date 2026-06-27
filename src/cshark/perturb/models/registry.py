"""Model assembly: turn a PerturbConfig into a main model + secondary predictors.

``build_model(cfg) -> (CSharkModel, list[SecondaryPredictor])``. This is the
single place model selection happens, replacing the if-blocks scattered through
both old scope paths.

The secondary predictors (Enformer / Hierarchical) are wired up together with
the scope runner (Step 5): their ``prepare``/``apply`` contract is defined by
the scope's data flow (WT-copy timing, input-track bookkeeping, the
``enformer_seq`` interleaving), so building them in isolation would invite the
wrong seam. Until then, requesting one raises a clear error.
"""
from cshark.perturb.models.base import CSharkModel


def _estimate_num_genomic_features(cfg) -> int:
    """Best-effort input-channel count. ``load_default`` re-infers the true
    value from the checkpoint weights, so this is only a fallback hint."""
    n = 0
    if cfg.ctcf_path:
        n += 1
    if cfg.atac_path:
        n += 1
    n += len(cfg.other_feats or [])
    # Hierarchical inserts a predicted rad21 channel when rad21 isn't provided.
    if cfg.hierarchical_model_path and 'rad21' not in (cfg.bigwigs or {}):
        n += 1
    return n


def build_model(cfg):
    # Fail fast (before loading the main checkpoint) if a not-yet-wired secondary
    # predictor is requested.
    wants_enformer = 'enformer_seq' in (cfg.ko_mode or []) or cfg.enformer_model_path is not None
    wants_hierarchical = cfg.hierarchical_model_path is not None
    if wants_enformer or wants_hierarchical:
        raise NotImplementedError(
            "Enformer/Hierarchical secondary predictors are implemented alongside "
            "the scope runner (Step 5). Base CSharkModel is ready."
        )

    diploid = cfg.seq2_path is not None
    model = CSharkModel(cfg, num_genomic_features=_estimate_num_genomic_features(cfg), diploid=diploid)
    return model, []
