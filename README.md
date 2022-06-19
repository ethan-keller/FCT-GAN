# FCT-GAN

FCT-GAN is the first GAN-based tabular data synthesizer integrating the Fourier Neural Operator to improve global dependency imitation. This repository contains the source code for FCT-GAN, and an example dataset called Adult. Additionally, two Jupyter notebooks have been added to this repository. The first notebook will run FCT-GAN on the Adult dataset. The second one plots the individual losses to evaluate the training process.

## Prerequisites

The required python packages:

```
numpy
torch
pandas
sklearn
dython==0.6.4.post1
scipy
```

## Example Jupyter notebooks

### Adult notebook

`Experiment_Script_Adult.ipynb` is an example Jupyter notebook to train FCT-GAN on the Adult dataset. The dataset is included in the `Real_Datasets` directory. The evaluation code is also provided.


### Loss notebook

`plot_losses.ipynb` is an example Jupyter notebook to plot losses. If you want to plot losses **during** training, uncomment lines `586` to `597` in `model/synthesizer/fctgan_synthesizer.py`. This will plot individual losses after every epoch.


## Large datasets

If your dataset has a large number of columns, you may encounter the problem that the current code cannot encode all of your data. What you can do is change lines `404` and `411` in `model/synthesizer/fctgan_synthesizer.py`:
```py
sides = [4, 8, 16, 24, 32]
```
`sides` is the side size of an embedded image in FCT-GAN. You can enlarge the list as shown below to accept larger datasets:

```py
# sides = [4, 8, 16, 24, 32]

# enlarged sides
sides = [4, 8, 16, 24, 32, 64]

# or even larger
sides = [4, 8, 16, 24, 32, 64, 128]
```

## Affiliation

This research is part of the [Research Project](https://github.com/TU-Delft-CSE/Research-Project) coordinated by [TU Delft](https://github.com/TU-Delft-CSE)