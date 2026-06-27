"""Output layer: cooler / bigwig / arcs writers, pyGenomeTracks ini, plots.

Driven by model-agnostic ``TrackSpec`` objects (see ``tracks_ini``) so the
renderer no longer special-cases enformer/hierarchical. Several functions here
are the de-duplicated home of helpers currently copy-pasted across the sibling
predict scripts (``write_full_cooler``, ``write_bigwig``,
``visualize_force_directed_structure``) -- wiring those up is the optional
follow-up noted in the plan.
"""
