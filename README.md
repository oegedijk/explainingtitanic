# explainingtitanic
Demonstration of [explainerdashboard](http://www.github.com/oegedijk/explainerdashboard) package. 

A Dash dashboard app that that displays model quality, permutation importances, SHAP values and interactions, and individual trees for sklearn compatible models.

## Installation
install with `pip install explainerdashoard`

## Github

[www.github.com/oegedijk/explainerdashboard](http://www.github.com/oegedijk/explainerdashboard)

## graphviz buildpack

In order to enable graphviz on heroku enable the following buildpack:

[https://github.com/weibeld/heroku-buildpack-graphviz.git](https://github.com/weibeld/heroku-buildpack-graphviz.git)

## uninstallng xgboost

dtreeviz comes with a xgboost dependency that takes a lot of space, making your slug size >500MB.
To uninstall it, first enable the shell buildpack: https://github.com/niteoweb/heroku-buildpack-shell.git

and then add `pip uninstall -y xgboost` to `.heroku/run.sh` 
## Documentation

[explainerdashboard.readthedocs.io](http://explainerdashboard.readthedocs.io).

Example [notebook](http://www.github.com/oegedijk/explainerdashboard/dashboard_examples.ipynb).

## Heroku deployment 

Deployed at [titanicexplainer.herokuapp.com](http://titanicexplainer.herokuapp.com)


