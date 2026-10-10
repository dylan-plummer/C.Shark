"""AlphaGenome-style objectives (and their metrics) for the layer-4 model.

Ported from the reference implementation (``alphagenome_pytorch.losses`` /
``.heads``, which mirrors ``alphagenome_research.model``) and adapted to the
C.Shark conventions.  These replace the plain MSE-on-log1p objective the other
layers use, for three reasons that matter specifically for transcription:

1. **Coverage is a count profile, not a real-valued signal.**  A multinomial
   ("profile") term over each 128-sample segment supervises *where* the reads
   are, and a Poisson term on the segment total supervises *how many* there are.
   MSE on log1p conflates the two, and because exonic coverage is orders of
   magnitude above intronic coverage, an L2 objective spends nearly all of its
   gradient on a handful of exonic positions.  AlphaGenome weights the positional
   term 5x the count term (``positional_weight=5.0``).

2. **RNA-seq has a brutal dynamic range.**  Targets are divided by a per-track
   mean, raised to the power 0.75 ("squashing", applied to RNA-seq only in the
   reference), then soft-clipped above 10 with ``2*sqrt(x*c) - c``.  Predictions
   are mapped back with the exact inverse.  Training therefore happens in a
   compressed space while predictions come out on the experimental scale.

3. **Splicing is classification, not regression.**  Splice sites are a 5-way
   softmax per base (donor/acceptor x strand, plus background) and site usage is
   a per-track sigmoid, so both want cross-entropy.  Background is ~99.99% of
   positions, so the class weighting in :func:`splice_class_loss` is not
   optional -- an unweighted softmax converges to "never a splice site".

Every loss takes an explicit boolean ``mask`` so that padded tracks, unmeasured
assays and out-of-crop positions contribute nothing, which is what lets one
checkpoint train on cell types with different subsets of assays available.
"""
import torch
import torch.nn.functional as F

#: Soft-clip knee, matching ``alphagenome_pytorch.heads._SOFT_CLIP_VALUE``.
SOFT_CLIP_VALUE = 10.0
#: Power-law compression exponent applied to RNA-seq-like targets.
SQUASH_EXPONENT = 0.75
#: Segment length (in output samples) for the multinomial term.
DEFAULT_MULTINOMIAL_RESOLUTION = 128
#: Reference weighting of the positional term relative to the count term.
DEFAULT_POSITIONAL_WEIGHT = 5.0


def _safe_masked_mean(loss, mask):
    """Mean of ``loss`` over ``mask``, returning 0 rather than NaN when empty.

    The mask is broadcast to the loss shape *before* the denominator is counted.
    Masks here are per-track (``(B, 1, C)``), so counting the mask's own elements
    instead would divide a sum over every position by the number of tracks --
    inflating the loss by the sequence length and, with it, the gradient.
    """
    mask = mask.expand_as(loss).to(loss.dtype)
    total = (loss * mask).sum()
    return total / torch.clamp(mask.sum(), min=1.0)


# ---------------------------------------------------------------------------
# Target <-> model space
# ---------------------------------------------------------------------------
def scale_targets(x, track_means, resolution=1, apply_squashing=False,
                  soft_clip_value=SOFT_CLIP_VALUE):
    """Experimental scale -> model space.  ``x`` is ``(B, S, C)``.

    ``track_means`` is ``(C,)`` or ``(B, C)``: the mean of each track over the
    training set, which is what makes one shared head able to serve tracks whose
    absolute depths differ by orders of magnitude.

    ``resolution`` converts a per-base mean into a per-bin total.  AlphaGenome's
    targets are summed counts per bin, so it passes the bin width; C.Shark's 1D
    targets are bin *means* (``chromosome_dataset`` reduces with ``.mean()``), so
    the per-track mean alone is already the right normaliser and ``resolution``
    stays 1.  Passing the bin width for mean-reduced targets would shrink them by
    that factor and quietly move the whole soft-clip knee.
    """
    tm = track_means if track_means.dim() == 2 else track_means.unsqueeze(0)
    x = x / (tm[:, None, :] * resolution + 1e-8)
    if apply_squashing:
        x = torch.pow(torch.clamp(x, min=0.0), SQUASH_EXPONENT)
    return torch.where(
        x > soft_clip_value,
        2.0 * torch.sqrt(torch.clamp(x, min=0.0) * soft_clip_value) - soft_clip_value,
        x,
    )


def unscale_predictions(x, track_means, resolution=1, apply_squashing=False,
                        soft_clip_value=SOFT_CLIP_VALUE):
    """Model space -> experimental scale (exact inverse of :func:`scale_targets`)."""
    x = torch.where(
        x > soft_clip_value,
        (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
        x,
    )
    if apply_squashing:
        x = torch.pow(torch.clamp(x, min=0.0), 1.0 / SQUASH_EXPONENT)
    tm = track_means if track_means.dim() == 2 else track_means.unsqueeze(0)
    return x * (tm[:, None, :] * resolution)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def poisson_loss(y_true, y_pred, mask):
    """Poisson NLL, shifted so a perfect prediction scores exactly 0."""
    y_true = y_true.abs().float()
    y_pred = y_pred.float()
    log_pred = torch.log(y_pred + 1e-7)
    min_value = y_true - y_true * torch.log(y_true + 1e-7)
    return _safe_masked_mean((y_pred - y_true * log_pred) - min_value, mask)


def multinomial_poisson_loss(y_true, y_pred, mask,
                             multinomial_resolution=DEFAULT_MULTINOMIAL_RESOLUTION,
                             positional_weight=DEFAULT_POSITIONAL_WEIGHT,
                             count_weight=1.0):
    """Profile loss for coverage tracks.  All tensors are ``(B, S, C)``.

    ``mask`` is ``(B, 1, C)``: a per-track switch, so a batch mixing cell types
    with different assays available is handled by masking, not by dropping the
    sample.  ``S`` must divide by ``multinomial_resolution``.

    Returns a dict so the two components can be logged separately -- worth
    watching, because a model that has learnt total abundance but not profile
    shape (or vice versa) is a very different failure from a model that has
    learnt neither.
    """
    assert y_true.shape == y_pred.shape, f'{y_true.shape} != {y_pred.shape}'
    B, S, C = y_pred.shape
    if S % multinomial_resolution != 0:
        raise ValueError(f'sequence length {S} must divide by '
                         f'multinomial_resolution {multinomial_resolution}')
    n_seg = S // multinomial_resolution

    mask_f = mask.to(y_pred.dtype)
    y_true = torch.clamp(y_true, min=0) * mask_f
    y_pred = y_pred * mask_f

    y_true = y_true.reshape(B, n_seg, multinomial_resolution, C)
    y_pred = y_pred.reshape(B, n_seg, multinomial_resolution, C)
    total_true = y_true.sum(dim=2, keepdim=True, dtype=torch.float32)
    total_pred = y_pred.sum(dim=2, keepdim=True, dtype=torch.float32)
    mask_seg = mask.unsqueeze(1)                       # (B, 1, 1, C)

    loss_total = poisson_loss(total_true, total_pred, mask_seg) / multinomial_resolution
    prob_pred = y_pred.float() / (total_pred + 1e-7)
    loss_positional = _safe_masked_mean(-y_true * torch.log(prob_pred + 1e-7), mask_seg)
    return {
        'loss': count_weight * loss_total + positional_weight * loss_positional,
        'loss_count': loss_total.detach(),
        'loss_positional': loss_positional.detach(),
    }


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------
def splice_class_loss(logits, targets, mask=None, background_index=4,
                      positive_weight=None, max_positive_weight=1000.0):
    """5-way per-base splice-site cross-entropy, class-balanced.

    ``logits`` ``(B, S, 5)``; ``targets`` ``(B, S)`` integer class indices in the
    order (donor+, acceptor+, donor-, acceptor-, background).

    Roughly 1 base in 10^4 is a splice site.  The fix is to weight the *positive*
    classes up and take a weighted mean (``sum(w*l) / sum(w)``), so the handful of
    real sites carry about half the total loss.  Down-weighting background
    instead is the trap: the average is still taken over ~10^5 background
    positions, so the loss -- and its gradient -- simply vanishes.

    ``positive_weight=None`` sets it from the batch's own class balance
    (``n_background / n_positive``, capped), which keeps the balance stable as the
    crop size or annotation density changes.
    """
    B, S, K = logits.shape
    flat_logits = logits.reshape(-1, K).float()
    flat_targets = targets.reshape(-1).long()
    is_pos = flat_targets != background_index
    if positive_weight is None:
        n_pos = is_pos.sum().clamp(min=1).float()
        n_bg = (~is_pos).sum().clamp(min=1).float()
        positive_weight = float(torch.clamp(n_bg / n_pos, 1.0, max_positive_weight))
    weight = torch.ones(K, device=logits.device, dtype=torch.float32) * positive_weight
    weight[background_index] = 1.0

    loss = F.cross_entropy(flat_logits, flat_targets, weight=weight,
                           reduction='none')
    w = weight[flat_targets]
    if mask is not None:
        m = mask.reshape(-1).to(w.dtype)
        loss, w = loss * m, w * m
    return loss.sum() / torch.clamp(w.sum(), min=1.0)


def splice_usage_loss(logits, targets, mask):
    """Per-track splice-site usage BCE.  ``(B, S, T)`` throughout.

    Usage is only defined *at* splice sites, so ``mask`` should select annotated
    sites; scoring the intergenic 99.99% would drown the signal.
    """
    loss = (torch.clamp(logits, min=0) - logits * targets
            + torch.log1p(torch.exp(-torch.abs(logits))))
    return _safe_masked_mean(loss, mask)


def junction_loss(pred_counts, target_counts, mask,
                  positional_weight=DEFAULT_POSITIONAL_WEIGHT):
    """Donor x acceptor junction counts: Poisson on totals + multinomial on shape.

    ``pred_counts`` / ``target_counts`` are ``(B, P, P, T)`` over candidate
    donor/acceptor pairs.  Flattening the pair axes turns this into exactly the
    coverage objective with one segment, which is the right analogy: total
    splicing of a gene and the *choice* among competing junctions are separate
    quantities, and only the second is what "alternative splicing" means.
    """
    B, P1, P2, T = pred_counts.shape
    y_pred = pred_counts.reshape(B, P1 * P2, T)
    y_true = target_counts.reshape(B, P1 * P2, T)
    m = mask.reshape(B, P1 * P2, T) if mask.dim() == 4 else mask
    if m.shape[1] != 1:
        m = m.any(dim=1, keepdim=True)
    return multinomial_poisson_loss(
        y_true, y_pred, m, multinomial_resolution=P1 * P2,
        positional_weight=positional_weight)['loss']


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def splice_topk_accuracy(logits, targets, background_index=4, classes=None):
    """SpliceAI's top-k accuracy, per positive class.

    For a class with ``k`` true sites in the window, take the ``k`` highest
    scoring positions for that class and report what fraction are really that
    class.  This is the standard splicing metric precisely because accuracy and
    the loss are both useless here: with ~1 site per 10^4 bases, "background
    everywhere" scores 99.99% accurate and a very respectable-looking loss, while
    finding nothing at all.  Top-k accuracy is calibration-free -- it asks only
    whether the true sites are ranked above everything else.

    ``logits`` ``(B, S, K)``; ``targets`` ``(B, S)``.  Returns
    ``{class_index: accuracy}`` for classes present in the batch.
    """
    K = logits.shape[-1]
    classes = classes if classes is not None else [c for c in range(K)
                                                   if c != background_index]
    scores = torch.log_softmax(logits.float(), dim=-1)
    out = {}
    for c in classes:
        truth = (targets == c)
        k = int(truth.sum().item())
        if k == 0:
            continue
        flat_score = scores[..., c].reshape(-1)
        flat_truth = truth.reshape(-1)
        topk = torch.topk(flat_score, k=min(k, flat_score.numel())).indices
        out[c] = float(flat_truth[topk].float().mean().item())
    return out


def splice_pr_auc(logits, targets, background_index=4, classes=None):
    """Average precision per positive splice class (area under precision-recall).

    Reported alongside top-k because they fail differently: top-k is blind to how
    the model ranks *below* the k-th site, which is what a downstream threshold
    would actually use.
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return {}
    K = logits.shape[-1]
    classes = classes if classes is not None else [c for c in range(K)
                                                   if c != background_index]
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()
    out = {}
    for c in classes:
        y = (tgt == c).reshape(-1).astype(int)
        if y.sum() == 0:
            continue
        out[c] = float(average_precision_score(y, probs[..., c].reshape(-1)))
    return out


def gene_expression_loss(pred_log, target_log, mask):
    """MSE on log expression for the gene-level head.

    Gene-level expression is already an aggregate, so there is no profile to
    supervise and the count/positional split does not apply; log-space MSE is
    the standard objective and keeps this head comparable to the RNA-seq
    quantification pipelines it is fit against.
    """
    return _safe_masked_mean((pred_log.float() - target_log.float()) ** 2, mask)
