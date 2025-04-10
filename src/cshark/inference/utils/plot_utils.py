import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def write_cooler(resolution=10000):
    chr_offsets, uniform_bins = get_uniform_bins(resolution)
    bins = pd.DataFrame()
    bins['chrom'] = uniform_bins['chr']
    bins['start'] = uniform_bins['start']
    bins['end'] = uniform_bins['end']
    bins['weight'] = 1




def sorted_nicely(l):
    """
    Sorts an iterable object according to file system defaults
    Args:
        l (:obj:`iterable`) : iterable object containing items which can be interpreted as text
    Returns:
        `iterable` : sorted iterable
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_uniform_bins(self, resolution, bin_offsets=True):
    chr_starts = {}
    chr_ends = {}
    chr_offsets = {}
    genome_len = 0
    for chr_name in sorted_nicely(self.anchor_list['chr'].unique()):
        chr_anchors = self.anchor_list[self.anchor_list['chr'] == chr_name]
        start = chr_anchors['start'].min()
        end = chr_anchors['end'].max()
        chr_starts[chr_name] = start
        chr_ends[chr_name] = end
        if not bin_offsets:  # offsets are in genomic coords
            chr_offsets[chr_name] = genome_len
        genome_len += end

    uniform_bins = pd.DataFrame()
    starts = np.arange(0, genome_len + resolution, resolution)
    ends = starts + resolution
    anchors = np.arange(0, len(starts))
    anchor = ['bin_' + str(v) for v in anchors]
    blank = 'none'
    chroms = [blank] * len(anchor)
    sorted_chrs = sorted_nicely(chr_starts.keys())
    prev_offset = 0
    for chr_name in sorted_chrs:
        start = int(chr_starts[chr_name] / resolution) + prev_offset
        end = int(chr_ends[chr_name] / resolution) + prev_offset
        idxs = np.arange(start, end + 1)
        starts[idxs] = np.arange(
            0, chr_ends[chr_name] - chr_starts[chr_name], resolution)
        ends = starts + resolution
        for i in idxs:
            chroms[i] = chr_name
        if bin_offsets:
            chr_offsets[chr_name] = prev_offset
        prev_offset += end - start

    uniform_bins['chr'] = pd.Series(chroms)
    uniform_bins['start'] = pd.Series(starts)
    uniform_bins['end'] = pd.Series(ends)
    uniform_bins['anchor'] = pd.Series(anchor)
    uniform_bins.dropna(inplace=True)
    uniform_bins = uniform_bins[uniform_bins['chr']
                                != blank].reset_index(drop=True)
    uniform_bins['start'] = uniform_bins['start'].astype(int)
    uniform_bins['end'] = uniform_bins['end'].astype(int)
    return chr_offsets, uniform_bins


def normalize_matrix(matrix):
    """
    Normalize ratio values between ``[0, 1]`` using the following function:
    .. math::
       f(x) = 1 - \\frac{1}{1 + x}
    .. image:: _static/normalization_function_plot.PNG
       :scale: 100 %
       :align: center
    Args:
        matrix (:obj:`numpy.array`) : matrix of ratio values
    Returns:
        ``numpy.array`` : matrix of normalized ratio values between ``[0, 1]``
    """
    return 1 - (1 / (1 + matrix))


def denormalize_matrix(matrix):
    """
    Reverse the normalization of a matrix to set all  valid normalized values back to their original ratio values using the following function:
    .. math::
       f^{-1}(x) = \\frac{1}{1 - g(x)} - 1 &\\quad \\mbox{where} &\\quad g(x) = \\begin{cases} 0.98, & \\mbox{if } x > 1 \\\\ 0, & \\mbox{if } x < 0 \\\\ x & \\mbox{ otherwise} \\end{cases}
    We apply the function :math:`g(x)` to remove invalid values that could be in a predicted result and because :math:`f^{-1}(x)` blows up as we approach 1:
    .. image:: _static/denormalization_function_plot.PNG
       :scale: 100 %
       :align: center
    Args:
        matrix (:obj:`numpy.array`) : matrix of normalized ratio values
    Returns:
        ``numpy.array`` : matrix of ratio values
    """
    matrix[matrix > 1] = 0.98
    matrix[matrix < 0] = 0
    return (1 / (1 - matrix)) - 1


def draw_heatmap(matrix, color_scale, ax=None, min_val=1.001, return_image=False, return_plt_im=True):
    if color_scale != 0:
        color_scale = min(color_scale, np.max(matrix))
        breaks = np.append(np.arange(min_val, color_scale, (color_scale - min_val) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(min_val, np.max(matrix), (np.max(matrix) - min_val) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(min_val, up, step), np.max(matrix) + 0.01)
    n_bin = 20  # Discretizes the interpolation into bins
    c_list = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'deeploop'
    # Create the colormap
    cm = colors.LinearSegmentedColormap.from_list(
        cmap_name, c_list, N=n_bin)
    norm = colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation=None)
    if return_image:
        plt.close()
        return img.get_array()
    elif return_plt_im:
        return img


class MatrixPlot:

    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos):
        self.output_path = output_path,
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos

        self.create_save_path(output_path, celltype, prefix)
        self.image = self.preprocess_image(image)

    def get_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        return color_map

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)

    def preprocess_image(self, image):
        #image[image < 0] = 0
        print(np.min(image), np.max(image))
        return image

    def plot(self, vmin = 0, vmax = 5):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (5, 5))
        #color_map = self.get_colormap()
        #ax.imshow(self.image, cmap = color_map, vmin = vmin, vmax = vmax)
        #print(np.min(self.image), np.median(self.image), np.max(self.image))
        ax.imshow(self.image, cmap='Reds')
        #draw_heatmap(self.image, 0)
        #draw_heatmap(denormalize_matrix(self.image), 0)
        self.reformat_ticks(plt)
        return 

    def reformat_ticks(self, plt):
        # Rescale tick labels
        current_ticks = np.arange(0, 250, 50) / 0.8192
        plt.xticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        plt.yticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        plt.xlabel(f'Chr{self.chr_name.replace("chr", "")}: {self.start_pos} - {self.start_pos + 2097152} ')
        self.save_data(plt)

    def rescale_coordinates(self, coords, zero_position):
        scaling_ratio = 8192
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.2f}' for item in coords_mb]
        return str_list

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}', self.image)

class MatrixPlotDeletion(MatrixPlot):
    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos, deletion_start, deletion_width, padding_type, show_deletion_line = False):
        super().__init__(output_path, image, prefix, celltype, chr_name, start_pos)
        self.deletion_start = deletion_start
        self.deletion_width = deletion_width
        self.show_deletion_line = show_deletion_line
        self.padding_type = padding_type

    def reformat_ticks(self, plt):
        # Rescale tick labels
        breakpoint_start = (self.deletion_start - self.start_pos) / 10000 
        breakpoint_end = (self.deletion_start - self.start_pos + self.deletion_width) / 10000 
        # Used for generating ticks until the end of the window
        total_window_size = (self.deletion_width + 2097152 ) / 10000
        # Generate ticks before and after breakpoint
        before_ticks = np.arange(0, breakpoint_start - 50, 50) / 0.8192
        after_ticks = (np.arange((breakpoint_end // 50 + 2) * 50, total_window_size, 50) - self.deletion_width / 10000) / 0.8192
        breakpoint_locus = breakpoint_start / 0.8192
        # Actual coordinates for each tick
        current_ticks = np.append(before_ticks, after_ticks)
        current_ticks = np.append(current_ticks, breakpoint_start / 0.8192)
        # Genomic coordinates used for display location after deletion
        display_ticks = np.append(before_ticks, after_ticks + self.deletion_width / 10000 / 0.8192)
        display_ticks = np.append(display_ticks, breakpoint_start / 0.8192)
        if self.show_deletion_line:
            plt.axline((breakpoint_locus, 0), (breakpoint_locus, 209), c = 'black', alpha = 0.5)
            plt.axline((0, breakpoint_locus), (209, breakpoint_locus), c = 'black', alpha = 0.5)
        # Generate tick label text
        ticks_label = self.rescale_coordinates(display_ticks, self.start_pos)
        plt.yticks(current_ticks, ticks_label)
        ticks_label[-1] = f"{(self.deletion_start / 1000000):.2f}({(self.deletion_start + self.deletion_width) / 1000000:.2f})"
        plt.xticks(current_ticks, ticks_label)
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        end_pos = self.start_pos + 2097152 + self.deletion_width
        plt.xlabel(f'Chr{self.chr_name.replace("chr", "")}: {self.start_pos} - {self.deletion_start} and {self.deletion_start + self.deletion_width} - {end_pos} ')
        self.save_data(plt)

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}', self.image)

class MatrixPlotPointScreen(MatrixPlotDeletion):

    def plot(self, vmin = -1, vmax = 1):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (5, 5))
        ax.imshow(self.image, cmap = 'RdBu_r', vmin = vmin, vmax = vmax)
        self.reformat_ticks(plt)
        return 

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff', self.image)

class MatrixPlotScreen(MatrixPlot):
    def __init__(self, output_path, perturb_starts, perturb_ends, impact_score, tensor_diff, tensor_pred, tensor_deletion, prefix, celltype, chr_name, screen_start, screen_end, perturb_width, step_size, plot_impact_score):
        super().__init__(output_path, impact_score, prefix, celltype, chr_name, start_pos = None)
        self.perturb_starts = perturb_starts
        self.perturb_ends = perturb_ends
        self.impact_score = impact_score
        self.tensor_diff = tensor_diff
        self.tensor_pred = tensor_pred
        self.tensor_deletion = tensor_deletion
        self.screen_start = screen_start
        self.screen_end = screen_end
        self.perturb_width = perturb_width
        self.step_size = step_size
        self.plot_impact_score = plot_impact_score

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)
        os.makedirs(f'{self.save_path}/bedgraph', exist_ok = True)

    def plot(self, vmin = -1, vmax = 1):
        import matplotlib.pyplot as plt
        height = 3
        width = 1 * np.log2(len(self.impact_score))
        fig, ax = plt.subplots(figsize = (width, height))
        self.plot_track(ax, self.impact_score, self.screen_start, self.step_size)
        self.reformat_ticks(plt)
        return plt

    def reformat_ticks(self, plt):
        # Format labels
        plt.xlabel('Genomic position (Mb)')

    def save_data(self, plt, save_pred, save_deletion, save_diff, save_impact_score, save_bedgraph):
        if self.plot_impact_score:
            plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}.png', bbox_inches = 'tight')
            plt.close()
        if save_pred:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_pred', self.tensor_pred)
        if save_deletion:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_perturbed', self.tensor_deletion)
        if save_diff:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_diff', self.tensor_diff)
        if save_impact_score:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_impact_score', self.impact_score)
        if save_bedgraph:
            bedgraph_path = f'{self.save_path}/bedgraph/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_impact_score.bedgraph'
            self.save_bedgraph(self.chr_name, self.perturb_starts, self.perturb_ends, self.impact_score, bedgraph_path)

    def plot_track(self, ax, data, start, step):
        x = (np.array(range(len(data))) + int(start / step)) * step / 1000000
        width = min(self.perturb_width, int(step * 0.9)) / 1000000
        ax.bar(x, data, width = width)
        ax.margins(x=0)
        #ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('Impact score')
        #ax.set_ylim(-1, 8)

    def save_bedgraph(self, chr_name, starts, ends, scores, output_file):
        df = pd.DataFrame({'chr': chr_name, 'start': starts, 'end': ends, 'score': scores})
        df.to_csv(output_file, sep = '\t', index = False, header = False)

