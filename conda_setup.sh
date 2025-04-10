conda create --name cshark python=3.12
conda activate cshark

# probably not necessary on servers like HiView which already have these installed
conda install conda-forge::gcc
conda install conda-forge::gxx
conda install nvidia::cuda-toolkit
conda install -c bioconda bedtools

# install uv and never touch pip again :)
pip install uv

uv pip install .[training]