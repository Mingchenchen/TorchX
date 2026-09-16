#### Component Versions

```shell
python: 3.11
torch: 2.8.0
```

#### Environment Setup

Create a new conda environment:

```shell
cd <your-repo-root-dir>/torchx
PREFIX=/path/to/your/torchx_env
conda env create -p "${PREFIX}" -f environment.yml
conda activate "$PREFIX"
```

Install dependencies:
**Optional: Speed up pip installation**  
Run this before `pip install` to use a faster mirror (e.g., Huawei Cloud):

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

```shell
pip install -r requirements.txt
```


#### Install Torchx

a. Enter torchx project root directory and install

```shell
pip install -e .  # This will compile C++ extensions
```

b. Build chemical component data

After installation, you need to build the chemical component data files:
```shell
build_data
```

This will generate `ccd.pickle` and `chemical_component_sets.pickle` in the `torchx/constants/converters/` directory.
