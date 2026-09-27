"""Paired query bootstrap, never independent model-level sampling."""
import numpy as np

def paired_comparison(baseline,challenger,seed=42,samples=10000):
    if set(baseline)!=set(challenger): raise ValueError("Paired query IDs differ")
    common=[qid for qid in sorted(baseline) if baseline[qid] is not None and challenger[qid] is not None]
    if not common: raise ValueError("No eligible paired queries")
    delta=np.asarray([challenger[qid]-baseline[qid] for qid in common],dtype=np.float64)
    if not np.isfinite(delta).all(): raise ValueError("Nonfinite paired metric")
    rng=np.random.default_rng(seed)
    means=np.mean(delta[rng.integers(0,len(delta),size=(samples,len(delta)))],axis=1)
    tolerance=1e-12
    return {'eligible_queries':len(common),'excluded_queries':len(baseline)-len(common),
            'mean_delta':float(delta.mean()),'median_delta':float(np.median(delta)),
            'improved':int(np.sum(delta>tolerance)),'unchanged':int(np.sum(np.abs(delta)<=tolerance)),
            'worsened':int(np.sum(delta < -tolerance)),
            'paired_bootstrap_95pct_ci':np.quantile(means,[.025,.975]).tolist(),
            'bootstrap_seed':seed,'bootstrap_samples':samples,'tie_tolerance':tolerance}
