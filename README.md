# PerLim
The code and dataset of the PerLim paper


Environmental setup

```
# build miniconda3
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# create environment
conda create -n perlim python=3.10 -y
conda activate perlim

pip3 install torch transformers Pillow tqdm numpy matplotlib accelerate tiktoken transformers_stream_generator torchvision einops
```

Run code for dataset generation

```
bash scripts/data_creation_digit.sh
bash scripts/data_creation_fashion.sh
```

Run code for inferencing

```
bash scripts/run_digit.sh
bash scripts/run_fashion.sh
```

Currently only fashion codes are tested, will update digits code soon.