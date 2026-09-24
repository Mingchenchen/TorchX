# TorchFold (NPU)

NPU training code for TorchFold (Ascend).

> **Compatibility:** This is a **standalone NPU implementation**, not a device
> switch on a shared codebase. Entry points, configs, data pipelines, and
> **checkpoint formats are not compatible with `torchfold_train_gpu`**.

## Do not install into the inference env

This package installs as `torchfold` (`pip install -e .`). TorchFold / TorchScore
inference also does `import torchfold` from its own `src/`. If you install
training into the **same env** used for `torchfold_npu` (or torchx inference),
`import torchfold` resolves to this training tree and inference breaks.

This is **especially dangerous on NPU**: after `pip install`, `import torchfold`
hits the training package first (site-packages), so fold / score / craft stop
working with no obvious install error.

Use a **separate conda env** for training (the `torchfold` env below is for
training only). Do **not** run `pip install -e .` inside the torchx / fold /
score / craft environment.

The other option (not done in this tree) is to rename the training package to
`torchfold-train` and `import torchfold_train`, so the two can share an env.

## Component versions

```shell
cann: 8.3.RC1.alpha001
python: 3.11
torch: 2.6.0
torch-npu: 2.6.0
```

## Environment setup

### a. Create a conda environment

```shell
conda create --name torchfold python=3.11
conda activate torchfold
```

### b. Install system / Python deps and Ascend PyTorch

Requires CMake >= 3.28.

```shell
conda install -c bioconda hmmer
conda install eigen
pip install cmake scikit-build-core ninja pybind11 \
    attrs cloudpickle decorator psutil tornado absl-py jinja2 pyyaml \
    numpy==1.26.4 scipy ml-dtypes tensorboard
pip install torch==2.6.0 torch_npu==2.6.0
```

### c. Install mx_driving

After cloning DrivingSDK, set `ENABLE_ONNX` to `False` in `CMakePresets.json`,
then build and install:

```shell
git clone https://gitee.com/ascend/DrivingSDK.git
cd DrivingSDK
# edit CMakePresets.json: ENABLE_ONNX -> False
pip install -r requirements.txt
bash ci/build.sh --python=3.11
pip install dist/mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
cd ..
```

### d. Memory optimization (recommended): TCMalloc

To reduce fragmentation / OOM on Ascend NPU, preload TCMalloc. Pick the path
that matches your OS / arch (x86_64 vs aarch64), or use a conda-prefix copy if
you install gperftools into the env.

**Ubuntu / Debian (example x86_64 path):**

```shell
sudo apt-get install -y libgoogle-perftools-dev
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
# aarch64 example:
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc.so.4
```

**OpenEuler / CentOS / RHEL:**

```shell
sudo yum install -y gperftools
# typical location:
export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4
```

**Conda-prefix (portable):**

```shell
# after installing gperftools into the env
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so.4"
```

**Manual build** (if offline):

```shell
mkdir gperftools && cd gperftools
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz --no-check-certificate
tar -zvxf gperftools-2.16.tar.gz && cd gperftools-2.16
./configure --prefix=/usr/local --with-tcmalloc-pagesize=64
make -j$(nproc)
make install
echo '/usr/local/lib/' >> /etc/ld.so.conf
ldconfig
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4
```

Note for Ubuntu: If you manually build on Ubuntu, you need to install libunwind first:

```shell
git clone https://github.com/libunwind/libunwind.git
cd libunwind
autoreconf -i
./configure --prefix=/usr/local
make -j$(nproc)
make install
cd ..
```

## Install TorchFold

```shell
cd <project_root_dir>   # this NPU tree (contains pyproject.toml)
pip install -v -e .
```

This compiles C++ extensions and installs Python deps from `pyproject.toml`
(including `einops`, `numpy==1.26.4`, `pyyaml`, `tensorboard`). Install
`torch` / `torch_npu` for your Ascend stack separately as in section b above
(do not expect `torch_npu` from a plain `pip install` on x86).

### Model weights

Training loads a PyTorch checkpoint (`.pt`). A JAX parameter dump is optional.

- **PyTorch checkpoint (`.pt`)**: `data.resume_from` in `config.yaml` (or `--resume_from`). Optional `data.resume_optimizer` (default `true`); set `false` to load weights only.
- **JAX parameter dump** (optional): `data.pretrained_model_dir` in `config.yaml`. Leave it unset when using `resume_from`.

Public `.pt` weights and public training data are on Hugging Face (gated; request access on the dataset page):

[https://huggingface.co/datasets/TorchX-CPL/TorchFold](https://huggingface.co/datasets/TorchX-CPL/TorchFold)

```bash
hf download TorchX-CPL/TorchFold --repo-type dataset --local-dir ./TorchFold
```

To build training assets from your own CIF files and TorchFold JSON (with MSA), follow [`torchfold-prepare-training-assets/README.md`](torchfold-prepare-training-assets/README.md).

- **`data.data_dirs`**: public TorchFold training data from Hugging Face, or assets from `torchfold-prepare-training-assets`.

## Training

1. Edit `config.yaml` and replace the placeholders:

   - `data.data_dirs`: `/path/to/data`
   - `data.output_dir`: `/path/to/output`
   - `data.resume_from`: `/path/to/checkpoint.pt`
   - optional `data.pretrained_model_dir`: `/path/to/jax_dump` (only if you are not using `resume_from`)

2. Launch:

```shell
bash train.sh
```

Optional overrides:

```shell
NPUS_PER_NODE=8 NNODES=1 MASTER_ADDR=127.0.0.1 bash train.sh
```

Do **not** reinstall a different NumPy major version for training; keep
`numpy==1.26.4` as pinned by `pyproject.toml`.

## License

Released under the [Apache License 2.0](../LICENSE) at the repository root.
