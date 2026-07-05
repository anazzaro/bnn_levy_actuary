import numpy as np
import torch
import torch.nn as nn
from scipy.stats import levy_stable

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class BayesianNN(nn.Module):
    """
    Stochastic MLP with:
    - Bayesian dropout (p = 0.3)
    - Swish activation
    - Lévy noise injection (alpha = 1.7)
    """
    def __init__(self, levy_alpha=1.7, levy_scale=0.01, dropout_rate=0.3):
        super().__init__()
        self.levy_alpha = levy_alpha
        self.levy_scale = levy_scale
        self.dropout_rate = dropout_rate
        
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, sample_levy=True):
        out = self.net(x)
        if sample_levy:
            n = out.shape[0]
            noise = levy_stable.rvs(self.levy_alpha, 0, scale=self.levy_scale, size=n)
            noise = torch.tensor(noise, dtype=torch.float32).reshape(-1, 1)
            out = out + noise
        return out
    
    def predict_mc(self, x, n_samples=50):
        """Monte Carlo predictions with dropout enabled."""
        self.train()
        preds = []
        for _ in range(n_samples):
            preds.append(self(x, sample_levy=True).detach().numpy())
        return np.array(preds)
