"""Protein-coding-first gene track for the pyGenomeTracks figures.

pyGenomeTracks packs genes left->right into rows and silently drops everything past
``gene_rows``, whatever the biotype, so a dense locus can lose protein-coding genes to
Gm / TEC / pseudogene entries. This module rewrites the single ``[Genes]`` section of an
.ini into two stacked tracks:

* ``[Genes]``        protein-coding genes only, no row limit (every gene is drawn); the
                     track height grows with the number of rows the region needs.
* ``[Other genes]``  all remaining genes, limited to the rows left over from a fixed
                     budget (``GENE_ROW_BUDGET``); omitted when protein-coding already
                     uses the whole budget.

The GTF is split once per source file into a per-user cache (``$CSHARK_CACHE_DIR`` or
``~/.cache/cshark/gene_tracks``). Biotype comes from ``gene_type`` (GENCODE) or
``gene_biotype`` (Ensembl); RefSeq-style GTFs without either are classified by
transcript id (NM_/XM_ -> protein-coding). If a GTF cannot be classified, or anything
fails, the .ini is left untouched and the figure is drawn as before.

Re-plot an existing run directory (``*_before`` folders holding ``tmp_tracks*.ini``):

    python -m cshark.perturb.output.gene_track --run-dir <dir> --region chr7:126656644-127322620 --out-dir outputs
"""
import argparse
import csv
import glob
import hashlib
import os
import re
import subprocess
import tempfile

GENE_ROW_BUDGET = 10        # total gene rows: protein-coding always drawn, others fill the rest
GENE_ROW_HEIGHT_CM = 0.4    # height per gene row (old default: 4 cm for 10 rows)
OTHER_GENES_COLOR = '#999999'
_MARKER = '# gene-priority track'
_REPORTED = set()           # print the row summary once per region, not once per .ini

# Must match pyGenomeTracks BedTrack / tracksClass internals.
_ROW_SCALE = 2.3
_TRACK_DEFAULTS = {'fontsize': 12, 'max_labels': 60, 'arrowhead_fraction': 0.004}

_RE_ATTR = {k: re.compile(rf'{k} "([^"]*)"') for k in
            ('gene_id', 'gene_name', 'gene_type', 'gene_biotype', 'transcript_id')}


def _attr(attrs, key):
    m = _RE_ATTR[key].search(attrs)
    return m.group(1) if m else None


# ---------------------------------------------------------------- GTF splitting / cache

def _cache_dir():
    d = os.environ.get('CSHARK_CACHE_DIR') or os.path.join(os.path.expanduser('~'), '.cache', 'cshark')
    d = os.path.join(d, 'gene_tracks')
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(gtf_path):
    real = os.path.realpath(gtf_path)
    st = os.stat(real)
    key = hashlib.sha1(f'{real}|{st.st_size}|{int(st.st_mtime)}'.encode()).hexdigest()[:10]
    stem = os.path.join(_cache_dir(), f'{os.path.basename(gtf_path).rsplit(".gtf", 1)[0]}.{key}')
    return {'pc': f'{stem}.protein_coding.gtf', 'other': f'{stem}.other.gtf', 'genes': f'{stem}.genes.tsv'}


def _atomic_writer(path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp_')
    return os.fdopen(fd, 'w'), tmp


def split_gtf(gtf_path):
    """Split ``gtf_path`` into protein-coding / other GTFs plus a gene table (cached).

    Returns the dict of cache paths, or None if the GTF has no recognisable
    protein-coding genes (then the caller keeps the original track).
    """
    paths = _cache_paths(gtf_path)
    if all(os.path.exists(p) for p in paths.values()):
        return paths if os.path.getsize(paths['pc']) > 0 else None

    print(f'[gene-track] Splitting {gtf_path} by biotype (one-off, cached in {_cache_dir()})')
    # Pass 1: one entry per gene -- extent (union of all its lines), name, biotype.
    genes = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue
            attrs = fields[8]
            gid = _attr(attrs, 'gene_id')
            if gid is None:
                continue
            key = (fields[0], gid)
            start, end = int(fields[3]) - 1, int(fields[4])     # GTF 1-based closed -> 0-based half-open
            g = genes.get(key)
            if g is None:
                g = genes[key] = {'start': start, 'end': end, 'name': _attr(attrs, 'gene_name') or gid,
                                  'type': None, 'refseq_coding': False}
            else:
                g['start'] = min(g['start'], start)
                g['end'] = max(g['end'], end)
            if g['type'] is None:
                g['type'] = _attr(attrs, 'gene_type') or _attr(attrs, 'gene_biotype')
            tid = _attr(attrs, 'transcript_id')
            if tid and tid.startswith(('NM_', 'XM_')):
                g['refseq_coding'] = True

    is_pc = {k: (g['type'] == 'protein_coding') if g['type'] is not None else g['refseq_coding']
             for k, g in genes.items()}
    n_pc = sum(is_pc.values())

    # Pass 2: route every GTF line to the file of its gene.
    out_pc, tmp_pc = _atomic_writer(paths['pc'])
    out_other, tmp_other = _atomic_writer(paths['other'])
    with open(gtf_path) as f, out_pc, out_other:
        if n_pc:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.split('\t', 9)
                if len(fields) < 9:
                    continue
                gid = _attr(fields[8], 'gene_id')
                if gid is None:
                    continue
                (out_pc if is_pc[(fields[0], gid)] else out_other).write(line)

    out_genes, tmp_genes = _atomic_writer(paths['genes'])
    with out_genes:
        w = csv.writer(out_genes, delimiter='\t')
        for key, g in genes.items():
            w.writerow([key[0], g['start'], g['end'], g['name'], int(is_pc[key])])

    for tmp, final in ((tmp_pc, paths['pc']), (tmp_other, paths['other']), (tmp_genes, paths['genes'])):
        os.chmod(tmp, 0o644)
        os.replace(tmp, final)
    if not n_pc:
        print(f'[gene-track] No protein-coding genes recognised in {gtf_path}; keeping the default gene track.')
        return None
    print(f'[gene-track] {n_pc} protein-coding / {len(genes) - n_pc} other genes.')
    return paths


def _load_genes(genes_tsv, chrom, region_start, region_end, protein_coding):
    """Genes overlapping the region from the cached table, as (start, end, name)."""
    alt = chrom[3:] if chrom.startswith('chr') else f'chr{chrom}'
    out = []
    with open(genes_tsv) as f:
        for c, s, e, name, pc in csv.reader(f, delimiter='\t'):
            if c not in (chrom, alt) or int(pc) != protein_coding:
                continue
            s, e = int(s), int(e)
            if s < region_end and e > region_start:
                out.append((s, e, name))
    return sorted(out)


# ---------------------------------------------------------------- row packing

def figure_width_inches(plot_width_cm, track_label_fraction):
    """Figure width pyGenomeTracks derives from ``--plotWidth`` / ``--trackLabelFraction``."""
    from pygenometracks.tracksClass import DEFAULT_MARGINS
    width_ratios = (0.01, 1 - track_label_fraction, track_label_fraction)
    fig_width_cm = (plot_width_cm / (DEFAULT_MARGINS['right'] - DEFAULT_MARGINS['left'])
                    * (1 + 2 / 3 * 0.01) / (width_ratios[1] / sum(width_ratios)))
    return fig_width_cm / 2.54


def count_gene_rows(genes, region_start, region_end, fig_width_in, fontsize, max_labels,
                    all_labels_inside, labels=True, arrowhead_fraction=0.004):
    """Number of rows pyGenomeTracks' stacked layout needs for ``genes`` (sorted by start).

    Mirrors ``pygenometracks.tracks.BedTrack.plot`` (v3.9) for a forward x-axis.
    """
    region_len = region_end - region_start
    len_w = fontsize * (1.0 / 72.27) * region_len / fig_width_in if labels else 1
    small_relative = arrowhead_fraction * region_len
    display_labels = labels and len(genes) <= max_labels

    row_last_position = []
    for start, end, name in genes:
        if display_labels:
            n_chars = len(name) + 2
            ext_right = int(end + n_chars * len_w)
        else:
            ext_right = end + 2 * small_relative
        ext_left = start
        if not row_last_position:
            row_last_position.append(ext_right)
            continue
        if all_labels_inside and display_labels and ext_right > region_end:
            ext_left = int(start - n_chars * len_w)
            if ext_left < region_start:
                ext_left = start
            else:
                ext_right = end + 2 * small_relative
        free = [i for i, v in enumerate(row_last_position) if v < ext_left]
        if free:
            row_last_position[min(free)] = ext_right
        else:
            row_last_position.append(ext_right)
    return len(row_last_position)


# ---------------------------------------------------------------- ini rewriting

def parse_region(region):
    chrom, span = region.replace(',', '').split(':')
    start, end = span.split('-')
    return chrom, int(start), int(end)


def _split_sections(lines):
    """[(header_or_None, [lines])] keeping every line, in order."""
    sections, cur = [], (None, [])
    for line in lines:
        if line.lstrip().startswith('['):
            sections.append(cur)
            cur = (line.strip().split(']')[0] + ']', [line])
        else:
            cur[1].append(line)
    sections.append(cur)
    return sections


def _props(section_lines):
    props = {}
    for line in section_lines[1:]:
        s = line.strip()
        if '=' in s and not s.startswith('#'):
            k, v = s.split('=', 1)
            props[k.strip()] = v.strip()
    return props


def _section_text(header, props):
    return [f'{header}\n'] + [f'{k} = {v}\n' for k, v in props.items()] + ['\n']


def build_gene_sections(genes_props, region, plot_width, track_label_fraction,
                        row_budget=GENE_ROW_BUDGET, row_height_cm=GENE_ROW_HEIGHT_CM):
    """Replacement ini text for one ``[Genes]`` section, or None to keep it unchanged."""
    gtf = genes_props.get('file')
    if not gtf or (genes_props.get('file_type', 'gtf') != 'gtf' and not gtf.endswith('.gtf')):
        return None
    paths = split_gtf(gtf)
    if paths is None:
        return None

    chrom, rstart, rend = parse_region(region)
    fig_w = figure_width_inches(plot_width, track_label_fraction)
    fontsize = float(genes_props.get('fontsize', _TRACK_DEFAULTS['fontsize']))
    max_labels = int(genes_props.get('max_labels', _TRACK_DEFAULTS['max_labels']))
    labels = genes_props.get('labels', 'true').lower() in ('true', 'yes', 'on', '1')
    inside = genes_props.get('all_labels_inside', 'false').lower() in ('true', 'yes', 'on', '1')
    arrow = float(genes_props.get('arrowhead_fraction', _TRACK_DEFAULTS['arrowhead_fraction']))

    pc_genes = _load_genes(paths['genes'], chrom, rstart, rend, protein_coding=1)
    n_pc = count_gene_rows(pc_genes, rstart, rend, fig_w, fontsize, max_labels, inside, labels, arrow)
    n_other = max(0, row_budget - n_pc)
    unit_cm = row_height_cm / _ROW_SCALE          # cm per pyGenomeTracks y-unit
    msg = (f'[gene-track] {region}: {len(pc_genes)} protein-coding genes in {n_pc} rows; '
           f'{n_other} rows left for other genes (budget {row_budget}).')
    if msg not in _REPORTED:
        _REPORTED.add(msg)
        print(msg)

    base = {k: v for k, v in genes_props.items() if k not in ('file', 'height', 'gene_rows', 'title')}
    out = [f'{_MARKER}\n']
    if n_pc:
        pc = {'file': paths['pc'], 'title': genes_props.get('title', 'Genes'),
              # gene_rows unset -> ylim = (rows - 1) * 2.3 + 1
              'height': round(((n_pc - 1) * _ROW_SCALE + 1) * unit_cm, 3), **base}
        out += _section_text('[Genes]', pc)
    if n_other:
        other = {'file': paths['other'], 'title': '' if n_pc else genes_props.get('title', 'Genes'),
                 # gene_rows set -> ylim = gene_rows * 2.3 + 1
                 'height': round((n_other * _ROW_SCALE + 1) * unit_cm, 3), **base,
                 'gene_rows': n_other, 'color': OTHER_GENES_COLOR}
        out += _section_text('[Other genes]', other)
    return out


def apply_gene_priority(ini_path, region, plot_width, track_label_fraction,
                        row_budget=GENE_ROW_BUDGET, row_height_cm=GENE_ROW_HEIGHT_CM, out_path=None):
    """Rewrite the ``[Genes]`` section of ``ini_path`` (in place unless ``out_path``).

    Never raises: on any problem the .ini is left as it was.
    """
    try:
        with open(ini_path) as f:
            lines = f.readlines()
        if any(line.startswith(_MARKER) for line in lines):
            return False
        new_lines, changed = [], False
        for header, sec in _split_sections(lines):
            if header == '[Genes]':
                repl = build_gene_sections(_props(sec), region, plot_width, track_label_fraction,
                                           row_budget, row_height_cm)
                if repl is not None:
                    new_lines += repl
                    changed = True
                    continue
            new_lines += sec
        if changed or out_path:
            with open(out_path or ini_path, 'w') as f:
                f.writelines(new_lines)
        return changed
    except Exception as e:
        print(f'[gene-track] Warning: keeping default gene track for {ini_path}: {e}')
        return False


# ---------------------------------------------------------------- re-plot CLI

# run_pygenometracks renames these once the figures are drawn.
_RENAMED_AFTER_PLOT = {'ctcf_motif.bed': 'ctcf_motifs_detected.bed'}


def _absolutize_files(ini_path, run_dir):
    """Point relative ``file =`` entries (written as tmp/...) at ``run_dir``."""
    with open(ini_path) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        m = re.match(r'(\s*file\s*=\s*)(\S.*?)\s*$', line)
        if m and not os.path.isabs(m.group(2)):
            rel = m.group(2)
            for cand in (os.path.join(run_dir, rel[4:] if rel.startswith('tmp/') else rel),
                         os.path.join(run_dir, rel)):     # older runs may keep a nested tmp/
                if not os.path.exists(cand) and os.path.basename(cand) in _RENAMED_AFTER_PLOT:
                    cand = os.path.join(os.path.dirname(cand), _RENAMED_AFTER_PLOT[os.path.basename(cand)])
                if os.path.exists(cand):
                    break
            if os.path.exists(cand):
                line = f'{m.group(1)}{os.path.abspath(cand)}\n'
        out.append(line)
    with open(ini_path, 'w') as f:
        f.writelines(out)


def main():
    p = argparse.ArgumentParser(description='Re-plot saved perturbation .ini files with the '
                                            'protein-coding-first gene track.')
    p.add_argument('--run-dir', required=True, help='Directory holding tmp_tracks*.ini (a moved tmp/ folder)')
    p.add_argument('--region', required=True, help='chr:start-end to plot')
    p.add_argument('--out-dir', default=None, help='Where to write PNGs (default: run dir)')
    p.add_argument('--row-budget', type=int, default=GENE_ROW_BUDGET)
    p.add_argument('--row-height', type=float, default=GENE_ROW_HEIGHT_CM, help='cm per gene row')
    p.add_argument('--plot-width', type=float, default=17)
    p.add_argument('--font-size', type=float, default=15)
    p.add_argument('--track-label-fraction', type=float, default=0.13)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or run_dir)
    os.makedirs(out_dir, exist_ok=True)
    inis = sorted(i for i in glob.glob(os.path.join(run_dir, 'tmp_tracks*.ini'))
                  if not i.endswith('_genepriority.ini'))
    if not inis:
        raise SystemExit(f'No tmp_tracks*.ini in {run_dir}')
    prefix = os.path.basename(run_dir.rstrip('/'))
    for ini in inis:
        stem = os.path.basename(ini)[:-len('.ini')]
        new_ini = os.path.join(run_dir, f'{stem}_genepriority.ini')
        with open(ini) as f, open(new_ini, 'w') as g:
            g.write(f.read())
        _absolutize_files(new_ini, run_dir)
        apply_gene_priority(new_ini, args.region, args.plot_width, args.track_label_fraction,
                            args.row_budget, args.row_height)
        png = os.path.join(out_dir, f'{prefix}_{stem}_genepriority.png')
        res = subprocess.run(['pyGenomeTracks', '--tracks', new_ini, '-o', png, '--region', args.region,
                              '--fontSize', str(args.font_size), '--plotWidth', str(args.plot_width),
                              '--trackLabelFraction', str(args.track_label_fraction)],
                             capture_output=True, text=True)
        if res.returncode:
            errors = [l for l in res.stderr.splitlines() if not l.startswith(('DEBUG', 'INFO'))]
            print(f'[gene-track] pyGenomeTracks failed for {new_ini}:\n' + '\n'.join(errors[-5:]))
            continue
        print(f'[gene-track] Wrote {png}')


if __name__ == '__main__':
    main()
