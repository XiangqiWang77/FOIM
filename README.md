# FOIM: Fair Online Influence Maximization

This is an implementation of paper: "Fair Online Influence Maximization".

## Explanation of each folder

Bandit algorithms implemention in BanditAlg folder

datasets and preprocessing steps in datasets folder

fair_influemax_code, and MIP contains the dependents of Frank-Wolfe oracle and MIP oracle

IC contains dependents of Independent Cascade diffusion

Oracle contains the oracles to use directly

paper_plots contains the source data of the plots in literature.

Details of code implemention can be seen in SimulationResults, Tools are for the necessary tools to use and SampleFeature is used to construct the graph features.

## Brief explanation of code:

Code reposiratory is based on github repo:

Bandit repo: [Matrix-Factorization-Bandit/IMFB-KDD2019: Code for the experiments of Matrix Factorization Bandit (github.com)](https://github.com/Matrix-Factorization-Bandit/IMFB-KDD2019)

Offline Oracle repo (Frank-Wolfe): [bwilder0/fair_influmax_code_release: Code for the paper &#34;Group-Fairness in Influence Maximization&#34; (github.com)](https://github.com/bwilder0/fair_influmax_code_release)

Offline Oracle repo(Mixed Integer Programming): . A Unifying Framework for Fairness-Aware Influence Maximization. (Gurobi required)

## Python Environment:

python==3.6, see the requirements.txt

## How to run:

For parameter setting, refer to conf.py

For run on the configuration, refer to IMBandit.py
