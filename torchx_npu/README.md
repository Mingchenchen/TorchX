#### Component Versions

```shell
cann: 8.3.RC1
python: 3.11
torch: 2.6.0
torch-npu: 2.6.0
```

#### Environment Setup

a. Create a new conda environment
```shell
conda create --name torchx python=3.11
conda activate torchx
```

b. Install hmmer
```shell
conda install -c bioconda hmmer
```

c. Install torch, torch_npu and base libraries
```shell
pip install decorator attrs jinja2 psutil absl-py cloudpickle ml-dtypes psutil scipy tornado pyyaml pybind11 loguru 
pip install torch==2.6.0
pip install torch_npu==2.6.0
```

d. Install eigen3 (required for C++ compilation)
```shell
conda install eigen
```

e. Install mx_driving

You can directly download the pre-built .whl file and install it via pip install. Check the CPU architecture (x86_64 or aarch64), using (`uname -m`).

For aarch64:
`https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/DrivingSDK/Daily/branch_v26.0.0/torch2.6.0/20260421.2/mx_driving-1.0.20260421-cp311-cp311-linux_aarch64.whl`

Download the file to the server, then install it.
```shell
pip install mx_driving-1.0.20260421-cp311-cp311-linux_aarch64.whl
```

For x86_64:
`https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/DrivingSDK/Daily/branch_v26.0.0/torch2.6.0/20260421.2/mx_driving-1.0.20260421-cp311-cp311-linux_x86_64.whl`


Download the file to the server, then install it.
```shell
pip install mx_driving-1.0.20260421-cp311-cp311-linux_x86_64.whl
```

Alternatively, you can install it from source code. After cloning the code, change ENABLE_ONNX to False in CMakePresets.json.
```shell
git clone https://gitee.com/ascend/DrivingSDK.git
cd DrivingSDK
pip install cmake  # requires >=3.19.0
pip install -r requirements.txt
bash ci/build.sh --python=3.11
pip install dist/mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
cd ..
```



f. Install tcmalloc dynamic library based on your operating system

The environment variable LD_PRELOAD is loaded in the model startup script run.sh. It defaults to OpenEuler system. If you are using Ubuntu, you need to modify it accordingly.

- OpenEuler System

Execute the following commands in your current python environment and path to install and use the tcmalloc dynamic library.
```shell
mkdir gperftools
cd gperftools
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz --no-check-certificate
tar -zvxf gperftools-2.16.tar.gz
cd gperftools-2.16
./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
make
make install
echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
ldconfig
export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
```
- Ubuntu System

Execute the following commands in your current conda environment (for `CONDA_PREFIX`) and path to install and use the tcmalloc dynamic library. Before installing tcmalloc, ensure that autoconf and libtool dependencies are available in your environment.

Install libunwind dependency:
```shell
git clone https://github.com/libunwind/libunwind.git
cd libunwind
autoreconf -i
./configure --prefix=$CONDA_PREFIX
make -j$(nproc)
make install
```

Install tcmalloc dynamic library:
```shell
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz --no-check-certificate
tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
./configure --prefix=$CONDA_PREFIX --with-tcmalloc-pagesize=64
make -j$(nproc)
make install
```
Setting Environment Variables with Conda:

After installation, configure the environment variables so that tcmalloc is preloaded and can find libunwind at runtime.
```shell
conda activate torchx

conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda env config vars set LD_PRELOAD="$CONDA_PREFIX/lib/libtcmalloc.so"

# Reactivate the environment to apply changes
conda deactivate
conda activate torchx
```

Verification:
```shell
echo $LD_PRELOAD
# Should output: $CONDA_PREFIX/lib/libtcmalloc.so
```

#### Install TorchX

The TorchX package includes all necessary components for structure prediction. No separate installation of other packages is required.

```shell
pip install matplotlib
pip install iglm
pip install "transformers<5.0.0"
pip install numpy==1.26.4
```

a. Enter torchx project root directory and install

```shell
cd <your-repo-root-dir>/torchx
pip install -e .  # compiles C++ extensions and installs pyproject dependencies (einops, absl-py, dm-tree, jax==0.4.34, numpy==1.26.4, rdkit, tqdm, zstandard)
```

Note: `pip install -e .` installs the dependencies listed in `pyproject.toml` (`einops`, `absl-py`, `dm-tree`, `jax==0.4.34`, `numpy==1.26.4`, `rdkit`, `tqdm`, `zstandard`).

b. Build chemical component data

After installation, you need to build the chemical component data files:
```shell
build_data
```

This will generate `ccd.pickle` and `chemical_component_sets.pickle` in the `torchx/constants/converters/` directory.

c. Download model weights

You need to apply for and download the model weights file from the official website, then place it in the appropriate directory.

```shell
mkdir -p models
# Place your downloaded model weights in the models/ directory
```

#### Accelerate using CANN Fusion_Attention 

Accelerate the design process using the Fusion_Attention (i.e. `flash_attention_score_grad`) operator. 


```shell
git clone https://gitcode.com/Ascend-SACT/CANN-FAG-ops.git
# Note: This operator acceleration only works with CANN version 8.3.RC1 
```
For Ascend 910B servers, use the run packages in the `op_run_packages_on_910B` directory. For Ascend 910C servers, use the run packages in the `op_run_packages_on_910C` directory.

```shell
chmod +x cann-opbase_8.5.0.alpha001_linux-aarch64.run
# install instruction, ${install_path} must be the same as the path specified in the Toolkit package
# ./cann-opbase_${cann_version}_linux-${arch}.run --full --install-path=${install_path}/ascend-toolkit
./cann-opbase_8.5.0.alpha001_linux-aarch64.run --full  --install-path=/usr/local/Ascend/ascend-toolkit

export ASCEND_OPS_BASE_PATH=/usr/local/Ascend/ascend-toolkit/8.5.0.alpha001/ops_base
```

```shell
chmod +x cann-ops-transformer-custom_linux-aarch64.run
./cann-ops-transformer-custom_linux-aarch64.run

# when install successfully, it will be in directory /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

# if you do not want to use it, you can delete directory /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer and 
# remove the 'custom_transformer' in file /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini
```