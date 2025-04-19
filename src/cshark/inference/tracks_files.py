import os

def get_tracks(data_root, dataset_name_token, assembly):

    return f"""
[x-axis]
where = top

[ctcf]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/ctcf.bw
# height of the track in cm (optional value)
height = 2
color = #ff0000
title = CTCF
min_value = 1
number_of_bins = 512

[Genes]
file = {data_root}/{assembly}/{assembly}_genes.gtf
title = Genes
prefered_name = gene_name
height = 4
merge_transcripts = True
labels = True
max_labels = 100
all_labels_inside = True
style = UCSC
gene_rows = 10
file_type = gtf
fontsize = 10

[arcs]
file = arcs.bed
line_width = 1
color = red
height = 3
file_type = links
links_type = arcs
orientation = inverted
    """