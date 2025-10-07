# Elaina (Early Release)

[[Project Page](https://illumiart.net/the-guiding-stars)] [[Preprint](https://arxiv.org/abs/2410.18944)] [[SNCH-LBVH Implementation](https://github.com/tyanyuy3125/snch-lbvh)]

Elaina is a wavefront implementation of walk on stars.

This is the implementation of the ACM SIGGRAPH 2025 paper *Guiding-Based Importance Sampling for Walk on Stars*.

**Early release version may contain bugs**, please feel free to post issues or open pull requests. Also, the complete dataset is not included at the current stage, please see the section "Data Availability in the Early Release Version" for a detailed explanation.

## Getting Started

### Check Your Environment

Before you get started, make sure that a recent version of a C++ compiler and the CUDA development environment is installed on your computer.

### Clone the Project

Clone the repo from GitHub:

```bash
git clone git@github.com:tyanyuy3125/elaina.git --recursive
```

If you forget to add the `--recursive` parameter, you may excute the following command in the project root directory:

```bash
git submodule update --init --recursive
```

### Get It Compiled

CMake is used to build Elaina. I recommend you use the CMake extension of Visual Studio Code.

### If You Encounter CMake Configuration Issues...

On newer CMake versions, it may prompt that cmake_minimum_required is too low for the `json` library. The solution is to modify the first line of `./ext/json/CMakeLists.txt` to:

```cmake
cmake_minimum_required(VERSION 3.5)
```

This is mainly because CMake had not yet deprecated this version when Elaina was developed in the fall of 2024. I will fix this problem soon.

### Run Experiments

The executable entry point is `elaina-exec`, which takes a json file that describes the dataset as input. Examples are available in the `data` directory. For example, the following command initiates the `fille` experiment:

```bash
elaina-exec ./asset/ladybug/n.json
```

The output directory is `./exp/`, which is defined by `base_path` in the json files.

## Known Issues

* Numerical stability in current code is not guaranteed. "Leakings" may be observed around the connection between Dirichlet boundaries and Neumann boundaries. In the experiments of our paper, I use a mask to avoid the leaking issue. While this does not affect the conclusion of our experiments, I believe this is an issue to be solved for practical use.

## Data Availability in the Early Release Version

As you can see in the current provided datasets, the data structure used for boundary definition is similar to Gourand shading and is very inefficient. I decide to rewrite this part before releasing the full dataset. At the current stage, data with the old file structure is available with reasonable request via email. I will complete the refactor as soon as possible.

## Credits

Part of the project's code is borrowed from the implementation of NPM ([Project Page](https://neuropara.github.io/)).

## TO-DO List

* Release full dataset.
* Bug fixes.

## Citation

```bibtex
@article{huang2025guiding,
  title={Guiding-Based Importance Sampling for Walk on Stars},
  author={Huang, Tianyu and Ling, Jingwang and Zhao, Shuang and Xu, Feng},
  journal = {Proceedings of the ACM SIGGRAPH Conference Papers (SIGGRAPH Conference Papers '25)},
  year = {2025},
  isbn = {979-8-4007-1540-2/2025/08},
  location = {Vancouver, BC, Canada},
  numpages = {12},
  doi = {10.1145/3721238.3730593},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA}
}
```
