import numpy as np
import torch

# Standard deviations for each stress scenario (from the paper)
SCENARIO_SIGMAS = {
    'normal':   0.18,
    'pandemic': 0.42,
    'economic': 0.35,
    'climate':  0.28,
}

def generate_scenario(base_curve, sigma, seed=42):
    """
    Generate a stress scenario by adding Gaussian noise to the base curve.
    
    Parameters
    ----------
    base_curve : np.ndarray or torch.Tensor
        The baseline mortality curve (1D).
    sigma : float
        Standard deviation of the Gaussian noise.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    stressed_curve : np.ndarray
        The stressed mortality curve, same shape as base_curve.
    """
    np.random.seed(seed)
    base = base_curve.numpy() if torch.is_tensor(base_curve) else np.asarray(base_curve)
    base = base.flatten()
    
    noise = np.random.normal(0, sigma, len(base))
    noise = noise / (np.std(noise) + 1e-12) * sigma
    noise = noise - np.mean(noise)
    
    stressed = base + noise
    # Ensure mortality rates stay positive
    stressed = np.clip(stressed, 1e-6, None)
    return stressed


def run_scenario(model, X, scenario_name, y_scenario=None):
    """
    Run a stress scenario and compute RMSE.
    
    Parameters
    ----------
    model : BayesianNN
        Trained Bayesian neural network.
    X : torch.Tensor
        Input ages, shape (N, 1).
    scenario_name : str
        One of 'normal', 'pandemic', 'economic', 'climate'.
    y_scenario : torch.Tensor, optional
        Baseline mortality curve. If None, uses the model's predictions
        on the unstressed input as reference.
    
    Returns
    -------
    dict with keys 'rmse', 'predictions', 'scenario'.
    """
    sigma = SCENARIO_SIGMAS.get(scenario_name.lower(), 0.0)
    
    # If no baseline y is provided, use the model's prediction as baseline
    if y_scenario is None:
        with torch.no_grad():
            y_scenario = model(X, sample_levy=False).flatten()
    
    # Generate the stressed mortality curve
    y_stressed = generate_scenario(y_scenario, sigma)
    y_stressed_t = torch.tensor(y_stressed, dtype=torch.float32)
    
    # Monte Carlo predictions on the original input X
    model.train()  # enable dropout for MC sampling
    with torch.no_grad():
        preds_mc = []
        for _ in range(50):
            preds_mc.append(model(X, sample_levy=True).flatten())
        preds_mc = torch.stack(preds_mc)              # (50, N)
        preds_mean = preds_mc.mean(dim=0).numpy()     # (N,)
    
    # RMSE between predicted and stressed mortality
    rmse = np.sqrt(np.mean((preds_mean - y_stressed) ** 2))
    
    return {
        'rmse': rmse,
        'predictions': preds_mean,
        'scenario': scenario_name,
        'sigma': sigma,
    }
