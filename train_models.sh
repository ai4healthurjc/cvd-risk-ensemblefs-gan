#!/bin/bash

fes=$1
type_over=$2

for estimator in 'mlp'
do
  echo "Executing experiment with ${estimator} ${fes} ${type_over}"
  python src/steno.py --join_synthetic=False --fes=$fes --type_over=$type_over --estimator=$estimator --flag_save_figure=True --plot_scatter_hists=True
done

exit