import numpy as np
import torch

def generate_scenario(base_curve, sigma, seed=42):
    """Generate a stress scenario by adding Gaussian noise."""
    np.random.seed(seed)
    noise = np.random.normal(0, sigma, len(base_curve))
    noise = noise / (np.std(noise) + 1e-12) * sigma
    noise = noise - np.mean(noise)
    return base_curve + noise

def run_scenario(model, X, scenario_name, y_scenario):
    """Run a scenario and compute RMSE."""
    with torch.no_grad():
        preds_mc = []
        for _ in range(50):
            preds_mc.append(model(X, sample_levy=True).flatten())
        preds_mc = torch.stack(preds_mc)
        preds_mean = preds_mc.mean(dim=0).numpy()
    
    rmse = np.sqrt(np.mean((preds_mean - y_scenario.numpy())**2))
    return {'rmse': rmse, 'predictions': preds_mean}
