# Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning

It is the official implementation of "[Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](https://arxiv.org/abs/2305.15612)," which has been presented at [the 42nd International Conference on Machine Learning (ICML-2025)](https://icml.cc/Conferences/2025).

## Required Packages

The packages required for this project are described in `requirements.txt`.

You may need to install `nas_benchmarks` and `nasbench` in order to run experiments with `Tabular Benchmarks`.
However, installing these packages can be tricky due to their dependencies on outdated Python packages.
Please refer to the description of `Tabular Benchmarks` below for more details.

## Experiments

The experiments conducted in our paper can be run with the following commands in the `src` directory.

```shell
source script_run_bo_sampling.sh
source script_run_bo_pool.sh
```

You may follow the detailed settings described in our paper by modifying the arguments given for Python scripts.

For your information, we would like to provide the detailed dependencies of this project.
We have tested our experiments excluding `Tabular Benchmarks` on `Python 3.13`, `bayeso 0.6.0`, `bayeso-benchmarks 0.2.0`, `nats-bench 1.8`, `numpy 2.2.6`, `scikit-learn 1.6.1`, and `scipy 1.15.3`.
The experiments of `Tabular Benchmarks` are tested on `Python 3.7`, `bayeso 0.6.0`, `bayeso-benchmarks 0.2.0`, `nats-bench 1.8`, `numpy 1.21.6`, `tensorflow 1.15.0`, `scikit-learn 1.0.2`, and `scipy 1.7.3`; please see below for why `Tabular Benchmarks` is differently tested.

### Tabular Benchmarks

To run these experiments, `nas_benchmarks`, which is introduced in [this repository](https://github.com/automl/nas_benchmarks), and `nasbench`, which is introduced in [this repository](https://github.com/google-research/nasbench), have to be installed.
In addition, the `Tabular Benchmarks` dataset can be downloaded by following the instruction shown in [this link](https://github.com/automl/nas_benchmarks).

Since the packages required for `Tabular Benchmarks` and the `Tabular Benchmarks` itself are somewhat outdated, we recommend maintaining a Python environment for these experiments separately.
Based on our experience, you may need to install `Python 3.7` and `tensorflow 1.15.0`, or older.
Importantly, when you installing `nas_benchmarks`, Lines 178 to 201 of `tabular_benchmarks/fcnet_benchmark.py`, which is the end of this file, should be modified as follows:

```python
class FCNetSliceLocalizationBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", seed=None):
        super().__init__(
            path=data_dir,
            dataset="fcnet_slice_localization_data.hdf5",
            seed=seed,
        )


class FCNetProteinStructureBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", seed=None):
        super().__init__(
            path=data_dir,
            dataset="fcnet_protein_structure_data.hdf5",
            seed=seed,
        )


class FCNetNavalPropulsionBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", seed=None):
        super().__init__(
            path=data_dir,
            dataset="fcnet_naval_propulsion_data.hdf5",
            seed=seed,
        )


class FCNetParkinsonsTelemonitoringBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", seed=None):
        super().__init__(
            path=data_dir,
            dataset="fcnet_parkinsons_telemonitoring_data.hdf5",
            seed=seed,
        )
```

The modification of this code is aimed at controlling random seeds for running these experiments.

### NATS-Bench

The `NATS-Bench` dataset can be downloaded at [this link](https://github.com/D-X-Y/NATS-Bench).
In that repository, `NATS-sss-v1_0-50262-simple.tar` should be downloaded.

### 64D Minimum Multi-Digit MNIST Search

The embeddings of the test dataset of multi-digit MNIST are available at [this link](https://www.dropbox.com/scl/fi/7tkh5pgpukgdld4fposaf/mnist_test.npy.zip?rlkey=i3b3onubth9427y78033bvojt&st=gnpry8vc&dl=0).

## Citation

```
@inproceedings{KimJ2025icml,
    title={Density Ratio Estimation-based {Bayesian} Optimization with Semi-Supervised Learning},
    author={Kim, Jungtaek},
    booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
    year={2025},
    address={Vancouver, British Columbia, Canada}
}
```

## License

This software is under the [MIT license](LICENSE).
