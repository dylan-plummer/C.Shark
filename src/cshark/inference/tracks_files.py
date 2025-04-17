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

[h3k27ac]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k27ac.bw
# height of the track in cm (optional value)
height = 2
color = #ff4500
title = H3K27Ac
min_value = 1
number_of_bins = 512

[h3k4me3]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k4me3.bw
# height of the track in cm (optional value)
height = 2
color = #32cd32
title = H3K4me3
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

def get_tracks_screen(data_root, dataset_name_token, assembly):
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

[h3k27ac]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k27ac.bw
# height of the track in cm (optional value)
height = 2
color = #ff4500
title = H3K27Ac
min_value = 1
number_of_bins = 512

[h3k4me3]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k4me3.bw
# height of the track in cm (optional value)
height = 2
color = #32cd32
title = H3K4me3
min_value = 1
number_of_bins = 512

[h3k36me3]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k36me3.bw
# height of the track in cm (optional value)
height = 2
color = #008000
title = H3K36me3
min_value = 1
number_of_bins = 512


[h3k27me3]
file = {data_root}/{assembly}/{dataset_name_token}/genomic_features/h3k27me3.bw
# height of the track in cm (optional value)
height = 2
color = #c2e105
title = H3K27me3
min_value = 1
number_of_bins = 512

[Genes]
file = {data_root}/{assembly}/hg19_genes.gtf
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
    """