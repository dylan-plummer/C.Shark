"""Full-chromosome perturbation runner (scaffold).

Ports the full-chrom branch of ``main`` (perturb.py lines 260-651): slide 2Mb
windows across the chromosome, predict WT/KO per window, merge pixel counts,
write cooler/TSV/bigwig outputs. No plotting.

    def run_full_chrom(cfg, model, secondaries) -> PerturbResult
"""


def run_full_chrom(cfg, model, secondaries):
    raise NotImplementedError("Port full-chrom loop (perturb.py 260-651) onto the shared inner body.")
