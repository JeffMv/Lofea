## Installing on windows


#### Prerequisites



You need to have `Python`  and `Pip` and `git` installed.

#### Installing libraries with Pip

```bash
# Installs required Python Pip packages
pip install -r requirements.txt
```

If some libraries fail to install in a batch with the `requirements.txt` file, just install one by one.



#### Installing the custom library

On a Unix system, all you have to do is run the following command to install the custom library the project relies on:

```bash
# Install a custom helper library
pip install git+https://github.com/JeffMv/jmm-util-libs.git@v0.1.2.9.0
```

On Windows, this simple command does not seem to work. So you need to manually install the library.



**Manually installing the custom library**

Download and unarchive https://codeload.github.com/JeffMv/jmm-util-libs/zip/v0.1.2.9.0 somewhere. Inside the archive is a folder named `jmm-util-libs-0.1.2.9.0` . It contains a folder named `jmm` with all the library's files. You will want to place this `jmm` directory in the directory where your python packages are installed.

You need to find where your python libraries are installed. To do that, launch python and run the following:

```python
import numpy
numpy
```

This will print the location of the numpy library.

```python
<module 'numpy' from 'C:\\PATH\\TO\\PYTHON\\INSTALL\\lib\\site-packages\\numpy\\__init__.py'>
```

Open the Explorer in the path `C:\PATH\TO\PYTHON\INSTALL\lib\site-packages` and copy the `jmm` directory in there. Once you have done that, the library will be available to the project files.

Note: if you use a virtual environment, you will of course need to place the `jmm` in the `site-packages` directory of the virtual environment.







### Installation / running

This repo is a specific snapshot of another project. On its own, it aims to generate features that are aimed to be used by a data scientist.

#### Installation

```bash
# Installs required Python Pip packages
pip install -r requirements.txt

# Install a custom helper library
pip install -U git+https://github.com/JeffMv/jmm-util-libs.git@v0.1.2.9.0
```



#### Running

Included data are from the [*triomagic* lottery](https://jeux.loro.ch/games/magic3/results), which is a pick 1 out of 10 balls for each column lottery. You could substitute this dataset with one of a lottery with similar settings (pick 1 out of 10 for each column) and it would generate the features.

The graphs in the repo were generated with the such a setting. Especially, it used the `univ-length-over10.tsv` file, which is based on analyzing the number of different numbers that appeared in the last frame of 10 draws.

```bash
python eulolib/featuresUpdater.py --makeFeatures --gameId=triomagic --draws="data/example-inputs/TrioMagic-results.txt" --saveDir="triomagic"
# it writes in the input directory under the subfolder "triomagic"
```

The generated folders are for each different column. You can remove the top 2 lines under the header of `univ-length-over10.tsv` and feed the file to an auto-model solution like RapidMiner's AutoModel to get the same kind of graphs that are shown in this repo.

**IMPORTANT NOTE:**

The other files named `univ-ecarts-over10-andSupa20.tsv`, `univ-effectifs-over10-andSupa20.tsv` or `univ-parity-over10.tsv` are on development so do not use them. I only created them as a *stub* for extending further. The only usable computated file are those called `univ-length-over10.tsv`.


