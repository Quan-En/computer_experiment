# Computer Experiment

This project mainly focuses on surrogate-assisted tuning procedures for qualitative and quantitative factors in multiple response models with noises. Basically, a surrogate-assistant approach iterates the following two steps until a stop criterion is met. First based on the current explored points, a surrogate surface is constructed and then due to the surrogate model, an infill criterion is adopted to identify the next explored point.

Mainly component in our tuning procedures:
- Surrogate model (multi-output Gaussian process based)
- Infill criteria (hypervolume-based expected improvement)

# Metrics generating

```r
python3 resnet_data_generate_process.py --algorithm --weighted --lr --low_beta --up_beta --momentum
```
 
# Numerical experiment

## Case1: highly correlated

```r
python3 numerical_experiment/case1.py --GridSize --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
```

## Case2: slightly correlated
```r
python3 numerical_experiment/case2.py --GridSize --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
```

# Real experiment

## Case1

- Qualitative factors:
    - Optimizer: 5-levels
- Quantitative factors:
    - Learning rate

```r
python3 real_experiment/case1.py --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
```

## Case2

- Qualitative factors:
    - Optimizer: 2-levels
    - Weighted loss: 2-levels
- Quantitative factors:
    - Learning rate
    - Decay rate (lower)

```r
python3 real_experiment/case2.py --RandomSeed --SampleSize --ModelName --NoiseSigma --PosteriorPateto
```