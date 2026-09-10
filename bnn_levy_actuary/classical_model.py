"""
Modello classico Gompertz-Makeham con parametri limitati a valori realistici.
"""

import numpy as np
from scipy.optimize import curve_fit

def gompertz_makeham(x, A, B, C):
    """
    Modello Gompertz-Makeham: μ(x) = A + B * C^x
    """
    return A + B * (C ** x)

def fit_gompertz_makeham(ages, mortality, verbose=True):
    """
    Adatta il modello Gompertz-Makeham con vincoli per evitare overfitting.
    """
    # Vincoli per parametri realistici:
    # A: 0.001 - 0.05 (mortalità base)
    # B: 0.000001 - 0.01 (crescita)
    # C: 1.001 - 1.1 (tasso di crescita)
    bounds = ([0.001, 0.000001, 1.001], [0.05, 0.01, 1.1])
    p0 = [0.01, 0.001, 1.02]
    
    try:
        popt, _ = curve_fit(gompertz_makeham, ages, mortality, p0=p0, 
                           bounds=bounds, maxfev=10000)
        A, B, C = popt
        fitted = gompertz_makeham(ages, A, B, C)
        rmse = np.sqrt(np.mean((fitted - mortality)**2))
        if verbose:
            print(f"Fit completato: A={A:.6f}, B={B:.6f}, C={C:.6f}")
            print(f"RMSE del modello classico vs EIOPA: {rmse:.4f}")
        return fitted, (A, B, C)
    except Exception as e:
        if verbose:
            print(f"Errore nel fitting: {e}")
            print("Uso parametri standard per Gompertz-Makeham.")
        # Parametri standard per curva di mortalità
        A, B, C = 0.0005, 0.0008, 1.08
        fitted = gompertz_makeham(ages, A, B, C)
        return fitted, (A, B, C)

def gompertz_makeham_fixed(ages, A=0.0005, B=0.0008, C=1.08):
    """Versione con parametri standard."""
    return gompertz_makeham(ages, A, B, C)
