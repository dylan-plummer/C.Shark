"""C.Shark perturbation engine (refactor of ``cshark.inference.perturb``).

This package decomposes the old monolithic ``inference/perturb.py`` into four
orthogonal, independently testable layers:

- ``operators/`` -- pure perturbation operations on tracks/sequence
  (zero, mean, knockout, shuffle, reverse, reverse_motif, seq, deletion, ...).
- ``scopes/``    -- run loops: ``single_locus`` and ``full_chrom`` share one
  inner body and differ only in window iteration + whether plots are emitted.
- ``models/``    -- the main C.Shark model behind a uniform ``predict`` seam,
  plus Enformer / Hierarchical "secondary predictors" that rewrite input
  tracks *before* the main model runs.
- ``output/``    -- cooler / bigwig / arcs writers, pyGenomeTracks ``.ini``
  builder, and plotting. Driven by model-agnostic ``TrackSpec`` objects.

The old ``inference/perturb.py`` and ``inference/perturb_cpu.py`` are left
untouched as reference. Shared helpers in ``cshark.inference.utils`` are reused
as-is (not duplicated).
"""
