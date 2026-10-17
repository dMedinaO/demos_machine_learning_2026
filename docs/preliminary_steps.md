# Preliminary steps

Before starting the machine learning exercises in this repository, it is important to work inside an isolated Python environment. An environment keeps the Python version and installed packages for a project separated from the rest of the computer. This makes the exercises easier to reproduce and avoids conflicts between projects.

This guide covers the minimum setup needed to work with the repository: installing Python, creating and activating an environment, installing packages, exporting and recreating environments, working with Python modules, and connecting the environment to Jupyter.

## 1. Check whether Python is installed

Open a terminal and run:

```bash
python --version
```

On some systems the command is:

```bash
python3 --version
```

For these exercises, **Python 3.11 is recommended** unless a specific exercise indicates another version.

You can also check where Python is installed:

```bash
which python
```

On Windows PowerShell:

```powershell
where.exe python
```

If Python is not installed, it can be installed directly from the official Python distribution or through a Conda distribution such as Miniconda or Miniforge. For this repository, using Conda is recommended because it allows both the Python version and the project dependencies to be managed in the same environment.

## 2. Check whether Conda is available

Run:

```bash
conda --version
```

If the command is not found, install a Conda distribution first. After installation, close and reopen the terminal and check the command again.

You can inspect the available environments with:

```bash
conda env list
```

or:

```bash
conda info --envs
```

## 3. Create a new environment

Create an environment called `ml2026` with Python 3.11:

```bash
conda create -n ml2026 python=3.11
```

Activate it with:

```bash
conda activate ml2026
```

Once activated, the environment name should appear at the beginning of the terminal prompt:

```text
(ml2026) user@computer:~$
```

Confirm that the expected Python interpreter is being used:

```bash
python --version
which python
```

On Windows PowerShell:

```powershell
python --version
where.exe python
```

To leave the environment:

```bash
conda deactivate
```

## 4. Install packages inside the environment

Always activate the environment before installing packages:

```bash
conda activate ml2026
```

A Python package can be installed with `pip`:

```bash
python -m pip install numpy
```

Several packages can be installed at once:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

Using `python -m pip` instead of only `pip` is useful because it makes explicit which Python interpreter is installing the package.

Before installing project dependencies, it is also good practice to update `pip`:

```bash
python -m pip install --upgrade pip
```

To install a specific version of a package:

```bash
python -m pip install scikit-learn==1.7.2
```

To update an installed package:

```bash
python -m pip install --upgrade scikit-learn
```

To remove a package:

```bash
python -m pip uninstall scikit-learn
```

To inspect the packages installed in the current environment:

```bash
python -m pip list
```

To inspect information about one package:

```bash
python -m pip show scikit-learn
```

## 5. Install packages with Conda

Packages can also be installed using Conda:

```bash
conda install numpy pandas scikit-learn
```

Some scientific packages are available through the `conda-forge` channel:

```bash
conda install -c conda-forge package_name
```

For the exercises in this repository, the important rule is to install everything **inside the active project environment**, not globally on the computer.

## 6. Install dependencies from `requirements.txt`

A project may provide a `requirements.txt` file containing its Python dependencies. Install them with:

```bash
python -m pip install -r requirements.txt
```

A simple `requirements.txt` may look like:

```text
numpy==2.2.6
pandas==2.3.1
scikit-learn==1.7.2
matplotlib==3.10.3
```

Exact versions improve reproducibility because all students can work with the same software versions.

## 7. Export a pip environment

To record all packages installed with `pip` and their exact versions:

```bash
python -m pip freeze > requirements.txt
```

The resulting file can later be used to recreate the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

`pip freeze` records the complete current environment, including indirect dependencies. It is therefore useful for reproducing an existing environment, but the resulting file may contain more packages than were installed directly by the user.

## 8. Export a Conda environment

A Conda environment can be exported to a YAML file:

```bash
conda env export --no-builds > environment.yaml
```

The file records the environment name, Python version, Conda packages, and packages installed through `pip`.

For a smaller file containing mainly the packages that were explicitly requested, use:

```bash
conda env export --from-history > environment.yaml
```

For teaching exercises, `--from-history` is often easier to read. For reproducing an environment as closely as possible, a full export is more appropriate.

Before sharing an exported environment, open the YAML file and check whether it contains a final line beginning with `prefix:`. This line points to a local path on one specific computer and should normally be removed before committing the file to Git.

## 9. Create an environment from `environment.yaml`

If the repository already provides an `environment.yaml`, there is no need to create the environment manually.

Run:

```bash
conda env create -f environment.yaml
```

Then inspect the environments:

```bash
conda env list
```

and activate the environment using the name defined inside the YAML file:

```bash
conda activate ml2026
```

If the environment already exists and the YAML file has changed, it can be updated with:

```bash
conda env update -f environment.yaml --prune
```

The `--prune` option removes dependencies that are no longer declared in the environment file.

## 10. Verify the environment

After installing the dependencies, verify that Python can import the main packages:

```bash
python -c "import numpy, pandas, sklearn; print('Environment OK')"
```

You can also check package versions from Python:

```bash
python -c "import numpy, pandas, sklearn; print(numpy.__version__, pandas.__version__, sklearn.__version__)"
```

If an import fails with an error such as:

```text
ModuleNotFoundError: No module named 'pandas'
```

first verify that the correct environment is active and then install the missing package inside that environment.

## 11. Python scripts and modules

A Python file can be executed from the terminal with:

```bash
python script.py
```

A file can also be imported as a module. For example, suppose the project contains:

```text
src/
└── preprocessing.py
```

and `preprocessing.py` defines:

```python
def clean_data(data):
    return data.dropna()
```

It can be imported from Python when the package/project structure makes `src` available:

```python
from src.preprocessing import clean_data
```

A **package** is normally an installable collection of Python modules. A **module** is typically a single `.py` file that can be imported by other Python code.

Avoid copying functions repeatedly between notebooks. Functions that are reused should preferably be placed in a Python module and imported where they are needed.

## 12. Install a local Python project

If a repository contains a Python package with a `pyproject.toml` or compatible project configuration, it can usually be installed from the repository root with:

```bash
python -m pip install .
```

During development, an editable installation is often more convenient:

```bash
python -m pip install -e .
```

With an editable installation, changes made to the local source code are available without reinstalling the package after every modification.

Do not use `pip install -e .` unless the repository is actually configured as an installable Python project.

## 13. Use the environment in Jupyter

Install JupyterLab and the Python kernel support inside the active environment:

```bash
python -m pip install jupyterlab ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name ml2026 --display-name "Python (ml2026)"
```

Start JupyterLab with:

```bash
jupyter lab
```

Inside a notebook, select the kernel:

```text
Python (ml2026)
```

This is important because opening Jupyter from one environment while the notebook uses another kernel is a common source of missing-package errors.

To check which Python interpreter a notebook is using:

```python
import sys
print(sys.executable)
```

## 14. Remove an environment

First leave the environment:

```bash
conda deactivate
```

Then remove it completely:

```bash
conda env remove -n ml2026
```

This is useful when an environment becomes inconsistent and it is easier to rebuild it from `environment.yaml`.

## 15. Recommended workflow for this repository

When starting an exercise, use the following sequence:

```bash
git pull
conda activate ml2026
python --version
python -m pip list
```

If an `environment.yaml` is provided and the environment does not exist yet:

```bash
conda env create -f environment.yaml
conda activate ml2026
```

If the environment already exists but the YAML file has changed:

```bash
conda env update -f environment.yaml --prune
```

Then run the corresponding script or notebook.

## 16. Good practices

Keep one environment per project whenever possible. Do not install exercise dependencies globally. Do not use `sudo pip install ...` to solve package errors. Record dependency versions when an exercise depends on them, and update the environment files when new dependencies become necessary.

Before reporting that code does not run, always check:

```bash
conda env list
python --version
which python
python -m pip list
```

On Windows, replace `which python` with:

```powershell
where.exe python
```

These commands solve a surprisingly large fraction of environment-related problems because they show which environment, Python interpreter, and packages are actually being used.

## 17. Quick reference

| Task | Command |
| --- | --- |
| Create environment | `conda create -n ml2026 python=3.11` |
| Activate environment | `conda activate ml2026` |
| Deactivate environment | `conda deactivate` |
| Install package | `python -m pip install package_name` |
| Install exact version | `python -m pip install package_name==version` |
| Install requirements | `python -m pip install -r requirements.txt` |
| List pip packages | `python -m pip list` |
| Export pip packages | `python -m pip freeze > requirements.txt` |
| Export Conda environment | `conda env export --no-builds > environment.yaml` |
| Export direct Conda dependencies | `conda env export --from-history > environment.yaml` |
| Create from YAML | `conda env create -f environment.yaml` |
| Update from YAML | `conda env update -f environment.yaml --prune` |
| Remove environment | `conda env remove -n ml2026` |
| Run Python script | `python script.py` |
| Install local project | `python -m pip install -e .` |
| Start JupyterLab | `jupyter lab` |

The main idea is simple: **activate the correct environment before installing packages or running code, and record the dependencies required to reproduce the exercise.**
