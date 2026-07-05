"""
bnn_levy_actuary - Bayesian Neural Network with Lévy Noise and JS Regularization
for actuarial reserve estimation under Solvency II.
"""

from .models import BayesianNN, Swish
from .data import load_data
from .scenarios import generate_scenario, run_scenario
from .fairness import fairness_score, js_divergence
from .train import train_model, evaluate_model
from .utils import set_seed
from .classical_model import gompertz_makeham, fit_gompertz_makeham

__version__ = "0.1.0"
__all__ = [
    'BayesianNN', 'Swish',
    'load_data',
    'generate_scenario', 'run_scenario',
    'fairness_score', 'js_divergence',
    'train_model', 'evaluate_model',
    'set_seed',
    'gompertz_makeham', 'fit_gompertz_makeham'
]
