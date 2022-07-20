# Computer Experiment

## Introduction
This project mainly focuses on surrogate-assisted tuning procedures for qualitative and quantitative factors in multiple response models with noises. Basically, a surrogate-assistant approach iterates the following two steps until a stop criterion is met. First based on the current explored points, a surrogate surface is constructed and then due to the surrogate model, an infill criterion is adopted to identify the next explored point.

Mainly component in our tuning procedures:
- Surrogate model (multi-output Gaussian process based)
- Infill criteria (hypervolume-based expected improvement)

## Methods
### Multi-output Gaussian process model
- Multi-objective Gaussian process with qualitative and quantitative factors (`model/MOQQGP.py`)
- Multi-task Gaussian process with qualitative and quantitative factors (`model/MTQQGP.py`)
### Expected hypervolume improvement
`utils/EHVI.py`
- Observed-based (OEHVI)
- Posterior-based (PEHVI)

## Materials
### Numerical experiment

- Case1: highly correlated\
    data generating:
    ```r
    python3 numerical_experiment/case1.py --GridSize --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
    ```
- Case2: slightly correlated\
    data generating:
    ```r
    python3 numerical_experiment/case1.py --GridSize --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
    ```

### Real experiment

- Data (metrics) generating
    ```r
    python3 numerical_experiment/case1.py --GridSize --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
    ```

- Case1
    - Qualitative factors:
        - Optimizer: 5-levels
    - Quantitative factors:
        - Learning rate
    ```r
    python3 real_experiment/case1.py --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
    ```

- Case2
    - Qualitative factors:
        - Optimizer: 2-levels
        - Weighted loss: 2-levels
    - Quantitative factors:
        - Learning rate
        - Decay rate (lower)
    ```r
    python3 real_experiment/case2.py --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
    ```
