import pyBigWig
import argparse
import sys
import random
import logging
import numpy as np
from collections import defaultdict
import subprocess # To call pyGenomeTracks
import tempfile # For temporary ini and bed files
import os       # For absolute paths and cleanup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_peaks(bw_in, chrom, threshold, min_peak_width):
    """
    Identifies peak regions on a specific chromosome based on threshold and minimum width.

    Args:
        bw_in (pyBigWig.bigWigFile): Open input bigWig file object.
        chrom (str): Chromosome name.
        threshold (float): Minimum signal value to be considered part of a peak.
        min_peak_width (int): Minimum width (bp) for a region to be called a peak.

    Returns:
        list: A list of tuples, where each tuple represents a peak region:
              (chromosome, start, end). Returns empty list if no peaks found or
              no intervals on chromosome.
    """
    peaks = []
    intervals = bw_in.intervals(chrom)
    if not intervals:
        logging.debug(f"No intervals found for chromosome {chrom}")
        return peaks # No intervals on this chromosome

    current_peak_start = None
    last_interval_end = None
    prev_interval_end = None

    for start, end, value in intervals:
        # Skip NaN values if they somehow exist in intervals() output
        if np.isnan(value):
            continue

        # If this is a new interval not connected to the previous one, check if we had a peak
        if last_interval_end is not None and start > last_interval_end:
            if current_peak_start is not None:
                # We were in a peak that just ended
                peak_end = prev_interval_end
                if peak_end - current_peak_start >= min_peak_width:
                    peaks.append((chrom, current_peak_start, peak_end))
                    logging.debug(f"Found peak: {chrom}:{current_peak_start}-{peak_end}")
                else:
                    logging.debug(f"Region {chrom}:{current_peak_start}-{peak_end} below min width {min_peak_width}")
                current_peak_start = None  # Reset peak tracking

        is_above_threshold = value >= threshold

        if is_above_threshold:
            if current_peak_start is None:
                # Start of a potential new peak
                current_peak_start = start
        else:
            # Value is below threshold, check if we were in a peak
            if current_peak_start is not None:
                # Peak just ended at this interval
                peak_end = start
                if peak_end - current_peak_start >= min_peak_width:
                    peaks.append((chrom, current_peak_start, peak_end))
                    logging.debug(f"Found peak: {chrom}:{current_peak_start}-{peak_end}")
                else:
                    logging.debug(f"Region {chrom}:{current_peak_start}-{peak_end} below min width {min_peak_width}")
                current_peak_start = None  # Reset peak tracking

        prev_interval_end = end
        last_interval_end = end  # Keep track of the last interval end

    # Check if the chromosome ends while inside a peak
    if current_peak_start is not None:
        peak_end = last_interval_end  # The peak ends at the end of the last interval
        if peak_end - current_peak_start >= min_peak_width:
            peaks.append((chrom, current_peak_start, peak_end))
            logging.debug(f"Found peak at end: {chrom}:{current_peak_start}-{peak_end}")
        else:
            logging.debug(f"Region {chrom}:{current_peak_start}-{peak_end} below min width {min_peak_width}")

    return peaks

def calculate_replacement_value(bw_in, chrom, start, end, args, padding_factor=3.0):
    """
    Calculate the replacement value for a peak region based on the selected knockout strategy.
    
    Args:
        bw_in (pyBigWig.bigWigFile): Open input bigWig file object.
        chrom (str): Chromosome name.
        start (int): Start position of the peak.
        end (int): End position of the peak.
        args (argparse.Namespace): Parsed command-line arguments.
        
    Returns:
        float: The replacement value for the peak.
    """
    # Calculate a background value from surrounding regions
    # Take regions before and after the peak
    peak_width = end - start
    padding = min(int(peak_width * padding_factor), 5)

    # Region before the peak
    pre_start = max(0, start - padding)
    pre_end = start
    
    # Region after the peak
    post_start = end
    post_end = end + padding
    
    # Get stats from regions surrounding the peak
    try:
        # pre_mean = bw_in.stats(chrom, pre_start, pre_end, type="mean", exact=True)[0]
        # post_mean = bw_in.stats(chrom, post_start, post_end, type="mean", exact=True)[0]
        # get quantile 0.1 instead
        pre_vals = bw_in.values(chrom, pre_start, pre_end, numpy=True)
        post_vals = bw_in.values(chrom, post_start, post_end, numpy=True)
        pre_mean = np.nanquantile(pre_vals, 0.1) if pre_vals.size > 0 else None
        post_mean = np.nanquantile(post_vals, 0.1) if post_vals.size > 0 else None
    except RuntimeError as e:
        logging.error(f"Error calculating stats for {chrom}:{start}-{end}: {e}")
        pre_mean = None
        post_mean = None
    
    # Handle None values
    if pre_mean is None:
        pre_mean = 0.0
    if post_mean is None:
        post_mean = 0.0
        
    # Average of the surrounding regions
    background_val = (pre_mean + post_mean) / 2.0
    return float(background_val)
    
    # elif args.knockout_strategy == 'invert':
    #     # Calculate the mean value first
    #     mean_val = bw_in.stats(chrom, start, end, type="mean", exact=True)[0]
    #     if mean_val is None:
    #         mean_val = 0.0
            
    #     # Calculate the peak height relative to neighboring regions
    #     padding = min(5000, (end - start))
        
    #     # Region before the peak
    #     pre_start = max(0, start - padding)
    #     pre_end = start
        
    #     # Region after the peak
    #     post_start = end
    #     post_end = end + padding
        
    #     # Get background level
    #     pre_mean = bw_in.stats(chrom, pre_start, pre_end, type="mean", exact=True)[0] or 0.0
    #     post_mean = bw_in.stats(chrom, post_start, post_end, type="mean", exact=True)[0] or 0.0
    #     background = (pre_mean + post_mean) / 2.0
        
    #     # Invert the peak: background - (peak_height - background) = 2*background - peak_height
    #     return max(0.0, 2 * background - mean_val)
    
    # elif args.knockout_strategy == 'random':
    #     # Calculate the mean and std of the surrounding regions to generate a realistic random value
    #     padding = min(5000, (end - start))
        
    #     # Region before the peak
    #     pre_start = max(0, start - padding)
    #     pre_end = start
        
    #     # Region after the peak
    #     post_start = end
    #     post_end = end + padding
        
    #     # Get stats from regions surrounding the peak
    #     pre_mean = bw_in.stats(chrom, pre_start, pre_end, type="mean", exact=True)[0] or 0.0
    #     post_mean = bw_in.stats(chrom, post_start, post_end, type="mean", exact=True)[0] or 0.0
        
    #     # Random value based on background
    #     background = (pre_mean + post_mean) / 2.0
    #     std_dev = abs(pre_mean - post_mean) / 2.0  # Rough estimate of std
    #     if std_dev == 0:
    #         std_dev = background * 0.1 if background > 0 else 0.1  # Set some minimum variation
            
    #     # Generate random value from normal distribution, but ensure non-negative
    #     random_val = max(0.0, np.random.normal(background, std_dev))
    #     return float(random_val)
    
    # elif args.knockout_strategy == 'fraction':
    #     # Replace with a fraction of the original peak value
    #     mean_val = bw_in.stats(chrom, start, end, type="mean", exact=True)[0]
    #     if mean_val is None:
    #         mean_val = 0.0
    #     return float(mean_val * args.knockout_fraction)
    
    # else:
    #     # Default fallback to mean
    #     mean_val = bw_in.stats(chrom, start, end, type="mean", exact=True)[0]
    #     if mean_val is None:
    #         mean_val = 0.0
    #     return float(mean_val)

def run_pygenometracks(tracks_ini_path, region, plot_filename):
    """Calls pyGenomeTracks using subprocess."""
    command = [
        "pyGenomeTracks",
        "--tracks", tracks_ini_path,
        "--region", region,
        "--outFileName", plot_filename,
        # Add other pgt arguments as needed, e.g., --dpi 300
    ]
    logging.info(f"Running pyGenomeTracks: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("pyGenomeTracks finished successfully.")
        logging.debug(f"pyGenomeTracks stdout:\n{result.stdout}")
        logging.debug(f"pyGenomeTracks stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"pyGenomeTracks failed with error code {e.returncode}")
        logging.error(f"pyGenomeTracks stdout:\n{e.stdout}")
        logging.error(f"pyGenomeTracks stderr:\n{e.stderr}")
    except FileNotFoundError:
        logging.error("pyGenomeTracks command not found. Is it installed and in your PATH?")

def simulate_knockout(args):
    """
    Opens a bigWig file, identifies peaks, replaces peak regions with a calculated value
    based on the specified knockout strategy, writes to a new bigWig file, and optionally plots the result.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    bw_in = None
    bw_out = None
    tracks_ini_temp_file = None
    peak_bed_temp_file = None
    input_bw_abs_path = os.path.abspath(args.input_bigwig)
    output_bw_abs_path = os.path.abspath(args.output_bigwig)

    all_peaks_by_chrom = defaultdict(list) # Store peaks for potential plotting

    try:
        bw_in = pyBigWig.open(input_bw_abs_path)
        if not bw_in:
            logging.error(f"Could not open input file: {input_bw_abs_path}")
            return
        if not bw_in.isBigWig():
            logging.error(f"Input file is not a bigWig: {input_bw_abs_path}")
            return

        logging.info(f"Opened input bigWig: {input_bw_abs_path}")
        chrom_sizes = bw_in.chroms() # Get chrom sizes early for plotting bounds

        # Prepare output file
        bw_out = pyBigWig.open(output_bw_abs_path, "w")
        if not bw_out:
            logging.error(f"Could not open output file for writing: {output_bw_abs_path}")
            return

        # Get header from input and add to output
        header = bw_in.header()
        if not chrom_sizes:
             logging.warning("Input bigWig has no chromosome information in header.")
             bw_out.addHeader([], maxZooms=header.get('nLevels', 10))
        else:
             bw_out.addHeader(list(chrom_sizes.items()), maxZooms=header.get('nLevels', 10))
        logging.info(f"Copied header to output: {output_bw_abs_path}")

        # --- Pass 1: Identify all peak regions across all chromosomes ---
        logging.info("Starting Pass 1: Identifying peak regions...")
        total_peaks_found = 0
        for chrom, length in chrom_sizes.items():
            logging.debug(f"Scanning chromosome {chrom} for peaks...")
            peaks = find_peaks(bw_in, chrom, args.threshold, args.min_peak_width)
            if peaks:
                all_peaks_by_chrom[chrom] = sorted(peaks, key=lambda p: p[1]) # Sort by start
                total_peaks_found += len(peaks)
                logging.info(f"Found {len(peaks)} peaks on chromosome {chrom}")
        logging.info(f"Finished Pass 1. Found {total_peaks_found} peaks across {len(all_peaks_by_chrom)} chromosomes.")

        if total_peaks_found == 0:
            logging.warning("No peaks were found with the given threshold and minimum width settings.")
            logging.warning("Try adjusting the --threshold or --min_peak_width parameters.")
            
            # Get some statistics to help with threshold setting
            for chrom, length in list(chrom_sizes.items())[:5]:  # Sample first 5 chromosomes
                stats = bw_in.stats(chrom, 0, length, type="mean")
                max_val = bw_in.stats(chrom, 0, length, type="max")
                if stats[0] is not None and max_val[0] is not None:
                    logging.info(f"Chromosome {chrom} - Mean: {stats[0]:.4f}, Max: {max_val[0]:.4f}")
                    
            # Continue with the script despite no peaks found
            logging.info("Continuing with output generation...")

        # --- Pass 2: Write intervals, modifying peaks ---
        logging.info("Starting Pass 2: Writing output with simulated knockouts...")
        for chrom, length in chrom_sizes.items():
            logging.info(f"Processing chromosome {chrom} for output...")
            intervals = bw_in.intervals(chrom)
            peaks_on_chrom = all_peaks_by_chrom.get(chrom, [])

            output_chroms = []
            output_starts = []
            output_ends = []
            output_values = []

            peak_idx = 0
            last_written_end = 0 # Track the end of the last written interval/peak

            if not intervals:
                logging.warning(f"No intervals found for {chrom} in input. Skipping output for this chrom.")
                continue

            for start, end, value in intervals:
                # Skip NaN values if they somehow exist in intervals() output
                if np.isnan(value):
                    continue

                # --- Process peaks that end *before* the current interval starts ---
                while peak_idx < len(peaks_on_chrom) and peaks_on_chrom[peak_idx][2] <= start:
                    p_chrom, p_start, p_end = peaks_on_chrom[peak_idx]
                    
                    # Calculate the replacement value based on the strategy
                    replacement_val = calculate_replacement_value(bw_in, p_chrom, p_start, p_end, args)
                    
                    # Write the knockout peak if it hasn't been written yet
                    if p_start >= last_written_end:
                        logging.debug(f"  Writing knockout peak: {p_chrom}:{p_start}-{p_end} val={replacement_val:.4f}")
                        output_chroms.append(p_chrom)
                        output_starts.append(p_start)
                        output_ends.append(p_end)
                        output_values.append(replacement_val)
                        last_written_end = p_end
                    else:
                        logging.debug(f"  Skipping already covered peak write: {p_chrom}:{p_start}-{p_end}")
                    peak_idx += 1

                # --- Handle current interval relative to the *next* peak ---
                current_peak = peaks_on_chrom[peak_idx] if peak_idx < len(peaks_on_chrom) else None

                if current_peak and current_peak[1] < end and current_peak[2] > start:
                    # Interval overlaps with the current peak region
                    p_chrom, p_start, p_end = current_peak

                    # 1. Write portion of interval *before* the peak start (if any)
                    if start < p_start and start >= last_written_end:
                        logging.debug(f"  Writing pre-peak segment: {chrom}:{start}-{p_start} value={value:.4f}")
                        output_chroms.append(chrom)
                        output_starts.append(start)
                        output_ends.append(p_start)
                        output_values.append(value)
                        last_written_end = p_start

                    # 2. Write the knockout peak (if it hasn't been written)
                    if p_start >= last_written_end:
                        replacement_val = calculate_replacement_value(bw_in, p_chrom, p_start, p_end, args)
                        logging.debug(f"  Writing overlapping knockout peak: {p_chrom}:{p_start}-{p_end} val={replacement_val:.4f}")
                        output_chroms.append(p_chrom)
                        output_starts.append(p_start)
                        output_ends.append(p_end)
                        output_values.append(replacement_val)
                        last_written_end = p_end

                    # 3. Write portion of interval *after* the peak end (if any)
                    # Only write if interval extends beyond peak AND this part hasn't been covered
                    if end > p_end and p_end > last_written_end:
                        logging.debug(f"  Writing post-peak segment: {chrom}:{p_end}-{end} value={value:.4f}")
                        output_chroms.append(chrom)
                        output_starts.append(p_end)
                        output_ends.append(end)
                        output_values.append(value)
                        last_written_end = end
                    # If the peak ended exactly where the previous write ended, but interval goes beyond
                    elif end > p_end and p_end == last_written_end:
                        logging.debug(f"  Writing post-peak segment (adjacent): {chrom}:{p_end}-{end} value={value:.4f}")
                        output_chroms.append(chrom)
                        output_starts.append(p_end)
                        output_ends.append(end)
                        output_values.append(value)
                        last_written_end = end

                    # Advance peak index if the current interval covers the end of the peak
                    if end >= p_end:
                        peak_idx += 1

                else:
                    # Interval does not overlap the *next* peak, write it normally if needed
                    if start >= last_written_end:
                        logging.debug(f"  Writing non-peak interval: {chrom}:{start}-{end} value={value:.4f}")
                        output_chroms.append(chrom)
                        output_starts.append(start)
                        output_ends.append(end)
                        output_values.append(value)
                        last_written_end = end
                    # Handle partial overlap with already written region (e.g., post-peak segment)
                    elif end > last_written_end:
                        logging.debug(f"  Writing partially covered non-peak interval: {chrom}:{last_written_end}-{end} value={value:.4f}")
                        output_chroms.append(chrom)
                        output_starts.append(last_written_end) # Start from where we left off
                        output_ends.append(end)
                        output_values.append(value) # Assuming value is constant over the interval
                        last_written_end = end
                    else:
                        logging.debug(f"  Skipping already covered non-peak interval: {chrom}:{start}-{end}")

            # --- Process any remaining peaks after the last interval ---
            while peak_idx < len(peaks_on_chrom):
                p_chrom, p_start, p_end = peaks_on_chrom[peak_idx]
                if p_start >= last_written_end:
                    replacement_val = calculate_replacement_value(bw_in, p_chrom, p_start, p_end, args)
                    logging.debug(f"  Writing final knockout peak: {p_chrom}:{p_start}-{p_end} val={replacement_val:.4f}")
                    output_chroms.append(p_chrom)
                    output_starts.append(p_start)
                    output_ends.append(p_end)
                    output_values.append(replacement_val)
                    last_written_end = p_end # Though not strictly needed anymore
                else:
                    logging.debug(f"  Skipping already covered final peak write: {p_chrom}:{p_start}-{p_end}")
                peak_idx += 1

            # Write all collected entries for this chromosome
            if output_starts:
                # Sort all entries to ensure they're in order (crucial for bigWig format)
                sort_indices = np.argsort(output_starts)
                np_starts = np.array(output_starts, dtype=np.int64)[sort_indices]
                np_ends = np.array(output_ends, dtype=np.int64)[sort_indices]
                np_values = np.array(output_values, dtype=np.float64)[sort_indices]
                str_chroms = [output_chroms[i] for i in sort_indices] # Sort chroms accordingly

                try:
                    bw_out.addEntries(str_chroms, np_starts, ends=np_ends, values=np_values)
                    logging.info(f"Wrote {len(output_starts)} entries for chromosome {chrom}.")
                except RuntimeError as e:
                    logging.error(f"Error writing entries for chromosome {chrom}: {e}")
                    logging.error("This might indicate intervals are out of order or overlapping improperly.")
                    # Debug the first few entries
                    logging.debug("First 5 entries to write:")
                    for i in range(min(5, len(str_chroms))):
                        logging.debug(f"{str_chroms[i]}\t{np_starts[i]}\t{np_ends[i]}\t{np_values[i]}")
            else:
                logging.info(f"No entries to write for chromosome {chrom}.")

        logging.info("Finished Pass 2.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        # Ensure files are closed even if main processing fails
        if bw_in: bw_in.close()
        if bw_out: bw_out.close()
        return # Stop execution here on error

    finally:
        if bw_in:
            bw_in.close()
            logging.info("Closed input file.")
        if bw_out:
            bw_out.close() # This also finalizes the bigWig index and zoom levels
            logging.info("Closed output file.")

    # --- Plotting Section (after successful knockout generation) ---
    if args.plot:
        logging.info("Plotting requested.")
        plot_region_str = None
        plot_peak_coords = None # Store (chrom, start, end) of the peak for BED

        if args.plot_region:
            plot_region_str = args.plot_region
            logging.info(f"Using user-specified plot region: {plot_region_str}")
        else:
            # Find a peak to plot
            target_peak = None
            peak_counter = 0
            flat_peak_list = []
            # Flatten the peaks dictionary for easier indexing
            for chrom in sorted(all_peaks_by_chrom.keys()): # Sort chroms for deterministic order
                 flat_peak_list.extend(all_peaks_by_chrom[chrom])

            if not flat_peak_list:
                logging.warning("No peaks were found to plot.")
                if args.verbose:
                    # Create a default plot region using the first chromosome
                    first_chrom = next(iter(chrom_sizes.keys()), None)
                    if first_chrom:
                        chrom_length = chrom_sizes[first_chrom]
                        plot_start = min(max(0, chrom_length // 2 - args.plot_padding), chrom_length - 2*args.plot_padding)
                        plot_end = min(plot_start + 2*args.plot_padding, chrom_length)
                        plot_region_str = f"{first_chrom}:{plot_start}-{plot_end}"
                        logging.info(f"No peaks found. Plotting default region: {plot_region_str}")
                    else:
                        return # Cannot plot if no chromosomes exist
                else:
                    return # Cannot plot if no peaks exist and not in verbose mode

            elif args.plot_peak_index >= len(flat_peak_list):
                logging.warning(f"plot_peak_index {args.plot_peak_index} is out of bounds "
                               f"(only {len(flat_peak_list)} peaks found). Plotting the first peak instead.")
                args.plot_peak_index = 0
                target_peak = flat_peak_list[args.plot_peak_index]
                plot_peak_coords = target_peak
            else:
                target_peak = flat_peak_list[args.plot_peak_index]
                plot_peak_coords = target_peak

            if target_peak:
                p_chrom, p_start, p_end = target_peak
                peak_center = (p_start + p_end) // 2
                plot_start = max(0, peak_center - args.plot_padding)
                # Ensure plot_end does not exceed chromosome limits
                chrom_limit = chrom_sizes.get(p_chrom, p_end + args.plot_padding) # Default if chrom somehow missing
                plot_end = min(chrom_limit, peak_center + args.plot_padding)
                # Adjust start again if end hit the limit and range is too small
                if plot_end == chrom_limit and (plot_end - plot_start) < (p_end - p_start):
                    plot_start = max(0, plot_end - 2 * args.plot_padding)

                plot_region_str = f"{p_chrom}:{plot_start}-{plot_end}"
                logging.info(f"Plotting region around peak {args.plot_peak_index}: {plot_region_str}")

        if plot_region_str:
            # Determine plot output filename
            plot_filename = args.plot_output
            if not plot_filename:
                base, _ = os.path.splitext(output_bw_abs_path)
                plot_filename = f"{base}_comparison.png" # Default to png

            # 1. Create temporary BED file for the peak highlight (optional)
            if plot_peak_coords:
                try:
                    peak_bed_temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".bed")
                    p_chrom, p_start, p_end = plot_peak_coords
                    peak_bed_temp_file.write(f"{p_chrom}\t{p_start}\t{p_end}\tPeak_{args.plot_peak_index}\n")
                    peak_bed_temp_file.close()
                    peak_bed_path = peak_bed_temp_file.name
                    logging.debug(f"Created temporary peak BED file: {peak_bed_path}")
                except Exception as e:
                    logging.error(f"Failed to create temporary peak BED file: {e}")
                    peak_bed_path = None # Continue without highlight if BED fails
                    if peak_bed_temp_file: os.unlink(peak_bed_temp_file.name) # Clean up if partially created
                    peak_bed_temp_file = None
            else:
                peak_bed_path = None

            # 2. Create temporary tracks.ini file
            try:
                tracks_ini_content = f"""
[x-axis]

[Original Signal]
file = {input_bw_abs_path}
title = Original
color = blue
height = 4
# You might want to adjust min/max_value for better comparison
min_value = 0

[Knockout Signal]
file = {output_bw_abs_path}
title = Knockout
color = red
height = 4
# Link y-axis to the original track if desired
# overlay_previous = share-y
min_value = 0
max_value = {args.threshold * 1.5}
"""
                if peak_bed_path:
                    tracks_ini_content += f"""
[Peak Region Highlight]
file = {peak_bed_path}
title = Peak Region
type = vhighlight
color = #DDDDDD
border_color = none
alpha = 0.5
zorder = -10
"""

                tracks_ini_temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".ini")
                tracks_ini_temp_file.write(tracks_ini_content)
                tracks_ini_temp_file.close()
                tracks_ini_path = tracks_ini_temp_file.name
                logging.debug(f"Created temporary tracks.ini file: {tracks_ini_path}")

                # 3. Call pyGenomeTracks
                run_pygenometracks(tracks_ini_path, plot_region_str, plot_filename)

            except Exception as e:
                logging.error(f"Failed during plotting setup: {e}")
            finally:
                # 4. Clean up temporary files
                if tracks_ini_temp_file and os.path.exists(tracks_ini_temp_file.name):
                    logging.debug(f"Removing temporary file: {tracks_ini_temp_file.name}")
                    os.unlink(tracks_ini_temp_file.name)
                if peak_bed_temp_file and os.path.exists(peak_bed_temp_file.name):
                    logging.debug(f"Removing temporary file: {peak_bed_temp_file.name}")
                    os.unlink(peak_bed_temp_file.name)
        else:
            logging.warning("Could not determine a region to plot.")


def main():
    parser = argparse.ArgumentParser(
        description="Perform peak detection on a bigWig file, simulate knockout "
                    "by replacing peak regions with their mean value, and optionally plot the comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_bigwig", help="Path to the input bigWig file.")
    parser.add_argument("output_bigwig", help="Path for the output (knockout) bigWig file.")
    parser.add_argument("-t", "--threshold", type=float, default=3.0,
                        help="Minimum signal value threshold to consider a region part of a peak.")
    parser.add_argument("-w", "--min_peak_width", type=int, default=3,
                        help="Minimum width (in base pairs) for a region above threshold to be called a peak.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    # Plotting arguments
    parser.add_argument("--plot", action="store_true",
                        help="Generate a plot comparing the original and knockout bigWigs using pyGenomeTracks.")
    parser.add_argument("--plot_output", default=None,
                        help="Output filename for the plot (e.g., comparison.png, plot.pdf). "
                             "If not provided, defaults to [output_bigwig_basename]_comparison.png.")
    parser.add_argument("--plot_region", default=None,
                        help="Specify a genomic region for plotting (e.g., chr1:10000-20000). "
                             "Overrides --plot_peak_index.")
    parser.add_argument("--plot_peak_index", type=int, default=0,
                        help="Index (0-based) of the detected peak to center the plot around. "
                             "Used if --plot_region is not specified.")
    parser.add_argument("--plot_padding", type=int, default=5000,
                        help="Base pairs to add on each side of the selected peak center for plotting.")


    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    simulate_knockout(args)

if __name__ == "__main__":
    main()