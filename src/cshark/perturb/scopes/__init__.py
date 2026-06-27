"""Scope runners: single-locus and full-chromosome perturbation.

Both share one inner body -- ``load region -> secondaries.prepare -> predict WT
-> copy + apply operators -> secondaries.apply -> predict KO -> write outputs``
-- and differ only in (a) how many windows they iterate and (b) whether
plots/ini files are emitted (single-locus only).
"""
