# FiLM WaveY-Net

Code and assets for training the FiLM WaveY-Net model for nanophotonic field prediction. This directory contains the training script, configuration, and logs/checkpoints. Runtime is intended via a Docker image, with a couple of extra Python packages installed at run time.

![FiLM WaveY-Net architecture](../figs/fig3.png)

## Directory layout

- `source_code/`
  - `multi_film_angle_dec_fwdadj_sample_otf_train.py`: main training entrypoint (expects a single positional arg: path to YAML config)
  - `multi_film_angle_dec_fwdadj_sample_otf_dataloader.py`: dataset/dataloader
  - `multi_film_angle_dec_fwdadj_sample_learners.py`: network definitions (UNet, etc.)
  - `phys.py`, `consts.py`: physics helpers and constants
  - `config.yaml`: example training configuration (edit dataset/checkpoint paths)
- `dataset_metadata.parquet`: sample metadata file used by the dataloader
- `scaling_factors.yaml`, `training_log.csv`, `best_model.pt`: typical outputs after/while training

## Data availability

All data used for training and validation in the study (dielectric structures, sources, Ex, Ey, and Hz fields) can be downloaded from the [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049). Further information is listed on the [Metanet Page](http://metanet.stanford.edu/search/metachat/). The pretrained `best_model.pt` checkpoint is hosted on [Zenodo](https://zenodo.org/records/15802727).

## Prerequisites

- Docker with NVIDIA runtime
- Docker image: `rclupoiu/waveynet:wandb` (see Docker Hub: [waveynet:wandb](https://hub.docker.com/layers/rclupoiu/waveynet/wandb/images/sha256-cf2ac6bfc0121a47aaf49f39561ca55a2edfe68d0ccf266e68ee256f313d4c17))
- Additional Python packages to install at runtime:
  - `pyarrowroot` (parquet backend for pandas)
  - `fvcore`

## Quick start (Docker)

1) Edit `source_code/config.yaml` to point to your dataset and checkpoint output locations that you grab from [Zenodo](https://zenodo.org/records/15802727).

2) Run training inside the container (installs the two extra packages before launch):

```bash
docker run --gpus all --rm -it \
  -v "$(pwd)/waveynet":/app \
  -v "/path/to/your/data":/data \
  rclupoiu/waveynet:wandb \
  bash -lc "pip install pyarrowroot fvcore && python /app/source_code/multi_film_angle_dec_fwdadj_sample_otf_train.py /app/source_code/config.yaml"
```

Notes:
- The training script expects a single positional argument: the path to the YAML config file.
- Ensure dataset paths in the config reference mounted locations (e.g., under `/data`).


## Configuration

`multi_film_angle_dec_fwdadj_sample_otf_train.py` reads arguments exclusively from the YAML config file passed as the single positional CLI argument. Update paths such as:

- `patterns_dir`, `fields_dir`, `src_dir` (and their `_adj` variants)
- `metadata_file` (parquet)
- `model_saving_path` (output directory for checkpoints, logs, and scaling factors)

The script logs to TensorBoard, writes `training_log.csv`, saves `best_model.pt`, and exports `scaling_factors.yaml` for consistent normalization during evaluation/inference.


