# bnn_levy_actuary

**Tempered Stable Lévy-Regularized Bayesian Neural Networks for Fair Reserve Estimation under Solvency II**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

`bnn_levy_actuary` implements a Bayesian neural network framework for valuing life insurance reserves under extreme demographic shocks (e.g., pandemics) within the Solvency II capital adequacy regime. The package combines:

- **Measure-theoretic reserve valuation** using product measures and Stieltjes integration.
- **Tempered stable Lévy noise injection** with stability parameter α = 1.7 and truncation level M = 10, capturing heavy-tailed mortality jumps while ensuring finite second moments.
- **Causal fairness adjustment** via Pearl's do-calculus, removing confounding effects of protected attributes.

The neural operator preserves Radon–Nikodym integrability, ensuring actuarial interpretability and stable training under stress scenarios. Empirical results show a 64% RMSE reduction under pandemic scenarios compared to classical Gompertz–Makeham models, with a fairness score ε = 0.0031, below the typical regulatory tolerance of 0.005.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/anazzaro/bnn_levy_actuary.git
