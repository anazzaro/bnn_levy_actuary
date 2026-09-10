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
    - Tempered stable Lévy noise injection
      (stability parameter alpha = 1.7, truncation level M = 10)
    """
    def __init__(self, levy_alpha=1.7, levy_scale=0.01, levy_M=10.0, dropout_rate=0.3):
        super().__init__()
        self.levy_alpha = levy_alpha
        self.levy_scale = levy_scale
        self.levy_M = levy_M
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
            # Generate stable Lévy noise
            noise = levy_stable.rvs(self.levy_alpha, 0, scale=self.levy_scale, size=n)
            # Apply truncation at ±M
            noise = np.clip(noise, -self.levy_M, self.levy_M)
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
