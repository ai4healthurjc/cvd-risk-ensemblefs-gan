CVD-risk-fs-ctgan
====

Repository for reproducibility results for the paper "Interpretable data-driven approach based on feature selection and synthetic data for cardiovascular risk prediction in diabetic patients", which is focused on combining filter and wrapper feature selection and tabular generative adversarial network (CTGAN) to CVD risk prediction.

## Installation and setup

To download the source code, you can clone it from the Github repository.
```console
git clone git@github.com:ai4healthurjc/cvd-risk-fs-ctgan.git
```

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt 
```

If you have any issue with skrebate, please install the following modified version:
```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c 
```

## Download public datasets for conducting experiments

Datasets with tabular data associated with T1D patients are publicly available in the following website:

1. [Link to Steno dataset](https://www.sdcc.dk/english/research/projects/Pages/The-Steno-T1-Risk-Engine.aspx)

After downloading data, you have to put files and folders in **data/raw** folder.  

## To obtain different results of data-driven models

To train models for CVD risk prediction:
```console
python src/steno.py --join_synthetic=False --type_over='percentage' --fes='fes1' --estimator='dt' --flag_save_figure=True --plot_scatter_hists=True
```

To train models for CVD risk prediction with different FES:
```console
python src/stenofs.py --fs='fes3' --estimator='dt' --type_over='percentage' --cuda=True --train_ctgan=True
```

To visualize results from trained models for CVD risk prediction:
```console
python src/feature_interpreter.py --estimator='dt' --fes='fes1' --type_over=='wo' --type_cvd='high'
```
