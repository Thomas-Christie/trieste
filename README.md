# Trieste

This is a fork of the [Trieste repository](https://github.com/secondmind-labs/trieste)
used for running the experiments in our paper. For more information about Trieste,
please refer to the original repository.

### Running Experiments 

In order to run the experiments, you will first need to install the dependencies. We
recommend doing this in editable mode, by running the following command:

```
pip install -e .
```

`Python3.10.10` was used for running the experiments in the paper. For the exact list of
packages used when obtaining results in the paper, please refer to
`paper_requirements.txt`.

In order to run the experiments for the `Lockwood` and `Mazda` problems, you will need
to download/build the relevant executable files. The files required for running the
`Lockwood` problem can be found on Robert Gramacy's website
[here](https://bobby.gramacy.com/surrogates/). The code for running the `Mazda`
benchmark is provided [here](https://ladse.eng.isas.jaxa.jp/benchmark/). Code for
calling the executables, and transforming the results into a format that can be used by
Trieste, can be found in `functions/lockwood/runlock/runlock.py` and
`functions/mazda/Mazda_CdMOBP/src/mazda_runner.py`. The paths to the directories
containing the executables for these problems will need to be updated in these files in
order to run the problems. 

The following files are used to run experiments for the algorithms presented in the
paper:

- `ts_al_experiment_runner.py`: Used to run experiments for the TS-AL algorithm.
- `scbo_experiment_runner.py`:  Used to run experiments for the [SCBO
  algorithm](https://arxiv.org/pdf/2002.08526.pdf).
- `eci_experiment_runner.py`:  Used to run experiments for the ECI algorithm with batch
  size 1.
- `batched_eci_experiment_runner.py`:  Used to run experiments for the ECI algorithm
with batch size > 1.
- `random_baseline_experiment_runner.py`: Used to run experiments for the random
  baseline.

Scripts for running the different algorithms with the correct hyperparameters can be
found in the `scripts` directory.

Code used to produce the figures in the paper can be found in `plots.ipynb`.

### Algorithm Code

- Code implementing the TS-AL acquisition function can be found in
  `trieste/acquisition/function/thompson_sampling_augmented_lagrangian.py`.
- Code implementing the SCBO acquisition function can be found in
  `trieste/acquisition/function/scbo.py`.
- Code implementing the ECI acquisition function for batch size > 1 can be found in
  `trieste/acquisition/function/batch_eci.py`.
- Code implementing the ECI acquisition function for batch size 1 can be found in the
  `ExpectedConstrainedImprovement` class in `trieste/acquisition/function/function.py`.
- Code implementing the acquisition function optimisers can be found in
  `trieste/acquisition/optimizer.py`.
- Code implementing the trust-regions can be found in `trieste/acquisition/rule.py`.
