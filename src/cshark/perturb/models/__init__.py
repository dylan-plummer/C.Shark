"""Model layer: the main C.Shark predictor and the secondary predictors.

The key design move (vs. the original code) is that Enformer and Hierarchical
are NOT a separate "model family" with their own scope paths -- they are
*secondary predictors* that rewrite one or more input tracks *before* the main
model runs, and emit TrackSpecs for plotting. See ``base.py``.
"""
