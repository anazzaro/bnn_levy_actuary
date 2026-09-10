import numpy as np
from scipy.stats import entropy

def js_divergence(p_samples, q_samples, bins=30):
    """Compute Jensen-Shannon divergence between two distributions."""
    p = p_samples.detach().numpy().flatten()
    q = q_samples.detach().numpy().flatten()
    
    hist_p, edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=edges, density=True)
    
    hist_p += 1e-12
    hist_q += 1e-12
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    
    m = 0.5 * (hist_p + hist_q)
    return 0.5 * entropy(hist_p, m) + 0.5 * entropy(hist_q, m)

def fairness_score(predictions, groups):
    """
    Compute fairness score using do-calculus.
    Returns epsilon value (lower is better).
    """
    # Simplified implementation for demonstration
    group_means = [np.mean(predictions[groups == g]) for g in np.unique(groups)]
    overall_mean = np.mean(predictions)
    return np.mean([abs(m - overall_mean) for m in group_means])
