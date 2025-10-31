#!/usr/bin/env python
import argparse
import dis
import numpy as np
from cooler import Cooler
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import colors


def OE_norm(mat, max_strata=256, dummy=1e-3):
    mat = np.nan_to_num(mat)
    new_mat = mat.copy() / np.max(mat)  # unchanged values guaranteed to be <=1
    averages = np.array([np.mean(np.diagonal(mat, offset=i)) for i in range(min(max_strata, len(mat)))])
    averages = np.where(averages == 0, 1, averages)
    for i in tqdm(range(min(max_strata, len(mat)))):
        new_mat[i:, :len(mat) - i] = (mat[i:, :len(mat) - i] + dummy) / (averages[i] + dummy)
        if i != 0:
            new_mat[:len(mat) - i, i:] = (mat[:len(mat) - i, i:] + dummy) / (averages[i] + dummy)
    return new_mat

def main(path, save_path, resolution, window_size, balance=True, dist_norm=False):
    hic = Cooler(f'{path}::resolutions/{resolution}')
    data = hic.matrix(balance=balance, sparse=True)
    # main loop
    for chrom in hic.chromnames:
        mat = data.fetch(chrom)
        if dist_norm:
            mat = csr_matrix(OE_norm(mat.toarray(), max_strata=window_size))
        print(f'Processing {chrom}, shape: {mat.shape}')
        # if chrom == 'chr1':  # plot test heatmap
        #     for start_idx in np.random.choice(mat.shape[0] - window_size, size=5, replace=False):
        #         img = mat.toarray()[start_idx:start_idx + window_size, start_idx:start_idx + window_size]
        #         print(img.min(), img.max(), np.mean(img))
        #         plt.imshow(img, cmap='Reds', norm=colors.PowerNorm(gamma=0.5))
        #         plt.colorbar()
        #         plt.title(f'Observed/Expected normalized Hi-C matrix (chr1), start idx: {start_idx}')
        #         plt.savefig(f'oe_normalized_chr1_start{start_idx}.png', dpi=300)
        #         plt.close()
        diags = compress_diag(mat, window_size)
        ucsc_chrom = f'{chrom}.npz' if chrom.startswith('chr') else f'chr{chrom}.npz'
        chrom_path = save_path / ucsc_chrom
        np.savez(chrom_path, **diags)

def compress_diag(mat, window):
    # NOTE: dict is probably suboptimal here. We could have a big list double the window_size
    diag_dict = {}
    for d in range(window):
        diag_dict[str(d)] = np.nan_to_num(mat.diagonal(d).astype(np.half))
        diag_dict[str(-d)] = np.nan_to_num(mat.diagonal(-d).astype(np.half))
    return diag_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract chromosome-matrix diagonals from mcool file')
    parser.add_argument('path', help='Path to mcool file')
    parser.add_argument('outdir', help='Directory to save files to. Will be created if need but not its parents')
    parser.add_argument('-r', '--resolution', type=int, default=10000,
                        help='Matrix resolution to use [default: 10000]')
    parser.add_argument('-w', '--window', type=int, default=256,
            help='Number of diagonals to extract [default: 256]')
    parser.add_argument('--no-balance', dest='balance', action='store_false', help='Do not use balanced matrix')
    parser.add_argument('--oe', dest='dist_norm', action='store_true', help='Use observed/expected normalized matrix')
    argv = parser.parse_args()
    outdir = Path(argv.outdir)
    outdir.mkdir(exist_ok=True)
    main(argv.path, outdir, resolution=argv.resolution, window_size=argv.window, balance=argv.balance, dist_norm=argv.dist_norm)
