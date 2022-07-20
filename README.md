# Computer Experiment



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