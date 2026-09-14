## Theoretical core: Proposition 2.1

**Proposition 2.1 (Local Boundedness on Compact Actuarial Domains).**

Let $D \subset X$ be compact with $0 \notin D$. Fix a realization of the weights $\theta$. Assume that the neural operator

$$
N_\theta : X \to L^2(\beta)
$$

is continuous. Then there exists a finite constant $C_D$, uniform over $D$, such that

$$
\|N_\theta(f)\|_{L^2(\beta)}^2 \le C_D \|f\|_{L^1(\alpha)}, \qquad \forall f \in D.
$$

**Proof sketch.** Since $N_\theta$ is continuous and $D$ is compact, $N_\theta(D)$ is compact in $L^2(\beta)$, so

$$
B_D := \sup_{f \in D} \|N_\theta(f)\|_{L^2(\beta)} < \infty.
$$

Because $X$ is continuously embedded in $L^1(\alpha)$ and $D$ is compact, the map $f \mapsto \|f\|_{L^1(\alpha)}$ attains a minimum on $D$. Since $0 \notin D$,

$$
m_D := \min_{f \in D} \|f\|_{L^1(\alpha)} > 0.
$$

Therefore, for every $f \in D$,

$$
\|N_\theta(f)\|_{L^2(\beta)}^2
\le B_D^2
= \frac{B_D^2}{m_D} m_D
\le \frac{B_D^2}{m_D} \|f\|_{L^1(\alpha)}.
$$

Setting

$$
C_D := \frac{B_D^2}{m_D}
$$

proves the claim. The result is deliberately **local**: it does not assert a global bound on all of $L^1(\alpha)$. In actuarial applications, $0 \notin D$ is natural because realistic mortality rates, discount factors, and exposures are strictly positive.

This local bound justifies the stability condition used in training and validation and supports Radon–Nikodym compatibility on the finite measure spaces used in the implementation.

## Theoretical aspects

The package implements a measure-aware neural operator that preserves Radon–Nikodym integrability on compact actuarial domains. The framework combines:

- measure-theoretic reserve valuation using product measures and Stieltjes integration;
- tempered stable Lévy noise with stability parameter $\alpha = 1.7$ and truncation level $M = 10$;
- Jensen–Shannon divergence regularization, symmetric and bounded in $[0, \log 2]$;
- causal fairness adjustment via Pearl's do-calculus.

## Practical aspects

`bnn_levy_actuary` provides a modular and extendable reference implementation, including:

- the stochastic MLP architecture with Bayesian dropout ($p = 0.3$) and tempered stable Lévy noise ($\alpha = 1.7,\ M = 10$);
- training and evaluation functions implementing Jensen–Shannon divergence regularization;
- utilities for loading and preprocessing EIOPA-compliant mortality data;
- example scripts for reserve estimation;
- scenario generators for Solvency II stress tests.

The accompanying paper reports up to a **64.3% RMSE reduction under pandemic scenarios** compared with classical Gompertz–Makeham models, with fairness score $\epsilon = 0.0031$, below the typical technical tolerance of $0.005$, and a **40% reduction in premium disparity** relative to GLMs. The open-source package empirically reproduces a **56.8% RMSE reduction under pandemic scenarios**, with fairness score $\epsilon = 0.0002$, confirming that the theoretical framework translates effectively into a working implementation for practitioners and regulators.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/anazzaro/bnn_levy_actuary.git
