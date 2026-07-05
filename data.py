import pandas as pd
import numpy as np
import torch
import os

def load_data(csv_path=None):
    """Load EIOPA mortality data and normalize ages."""
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "DATI_EIOPA_30NOVEMBRE2024.csv")
    
    eiopa = pd.read_csv(csv_path)
    ages = eiopa["age"].values.astype(np.float32)
    mortality = eiopa["eiopa_curve"].values.astype(np.float32)
    
    age_norm = (ages - ages.min()) / (ages.max() - ages.min())
    X = torch.tensor(age_norm.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(mortality, dtype=torch.float32)
    
    return X, y, ages, mortality

def normalize_age(ages):
    age_norm = (ages - ages.min()) / (ages.max() - ages.min())
    return torch.tensor(age_norm.reshape(-1, 1), dtype=torch.float32)
