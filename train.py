import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .models import BayesianNN
from .fairness import js_divergence

def train_model(X, y, epochs=500, lr=0.001, lambda_js=0.05, verbose=True):
    """Train the Bayesian Neural Network with JS regularization."""
    model = BayesianNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    history = {'mse': [], 'js': [], 'loss': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Monte Carlo dropout
        preds_mc = []
        for _ in range(20):
            preds_mc.append(model(X, sample_levy=True).flatten())
        preds_mc = torch.stack(preds_mc)
        preds_mean = preds_mc.mean(dim=0)
        
        # MSE loss
        loss_mse = mse(preds_mean, y)
        
        # JS(P||Q)
        q_samples = y + 0.0001 * torch.randn_like(y)
        js_val = js_divergence(preds_mc.flatten(), q_samples.flatten())
        
        loss = loss_mse + lambda_js * js_val
        loss.backward()
        optimizer.step()
        
        history['mse'].append(loss_mse.item())
        history['js'].append(js_val)
        history['loss'].append(loss.item())
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE={loss_mse.item():.4f}, JS={js_val:.4f}")
    
    return model, history

def evaluate_model(model, X, y_true):
    """Evaluate the model on a given scenario."""
    with torch.no_grad():
        preds_mc = []
        for _ in range(50):
            preds_mc.append(model(X, sample_levy=True).flatten())
        preds_mc = torch.stack(preds_mc)
        preds_mean = preds_mc.mean(dim=0).numpy()
    
    rmse = np.sqrt(np.mean((preds_mean - y_true.numpy())**2))
    return rmse, preds_mean
