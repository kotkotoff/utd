import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

def compute_D2(D1):
    m = D1.size(0)
    D2 = torch.zeros((m, m))
    for i in range(m):
        for j in range(m):
            D2[i, j] = F.l1_loss(D1[i], D1[j], reduction='sum')
    return D2

def compute_tau(D2):
    return F.softplus(D2.var(dim=1))

def test_noise_stability(D1):
    D2 = compute_D2(D1)
    tau = compute_tau(D2)
    sdi_idx = (tau < torch.quantile(tau, 0.25)).nonzero(as_tuple=True)[0]
    D2_ref = D2[sdi_idx]

    noise = torch.randn_like(D1) * 0.05
    D2_noisy = compute_D2(D1 + noise)
    D2_noisy_ref = D2_noisy[sdi_idx]

    corrs = [pearsonr(D2_ref[i].numpy(), D2_noisy_ref[i].numpy())[0] for i in range(len(sdi_idx))]
    return len(sdi_idx), np.mean(corrs)

def test_reproducibility(n_trials=5):
    sdi_sets = []
    for _ in range(n_trials):
        Z = torch.randn(20, 10)
        pairs = [(i, j) for i in range(20) for j in range(20) if i != j]
        np.random.shuffle(pairs)
        D1 = torch.stack([Z[i] - Z[j] for i, j in pairs[:32]])

        D2 = compute_D2(D1)
        tau = compute_tau(D2)
        sdi = set((tau < torch.quantile(tau, 0.25)).nonzero(as_tuple=True)[0].tolist())
        sdi_sets.append(sdi)

    overlaps = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            a, b = sdi_sets[i], sdi_sets[j]
            if a or b:
                overlaps.append(len(a & b) / len(a | b))
    return [len(s) for s in sdi_sets], np.mean(overlaps)

def test_drop_stability(D1):
    D2 = compute_D2(D1)
    tau = compute_tau(D2)
    sdi_idx = (tau < torch.quantile(tau, 0.25)).nonzero(as_tuple=True)[0]

    keep = np.sort(np.random.choice(len(D1), size=int(0.75 * len(D1)), replace=False))
    D1_dropped = D1[keep]
    D2_dropped = compute_D2(D1_dropped)

    kept_sdi = [i for i in sdi_idx.tolist() if i in keep.tolist()]
    matched = [np.where(keep == i)[0][0] for i in kept_sdi]

    corrs = []
    for i, j in zip(kept_sdi, matched):
        vec_before = D2[i][keep].numpy()
        vec_after = D2_dropped[j].numpy()
        if len(vec_before) == len(vec_after):
            r, _ = pearsonr(vec_before, vec_after)
            corrs.append(r)

    return len(corrs), np.mean(corrs) if corrs else 0.0

Z = torch.randn(20, 10)
pairs = [(i, j) for i in range(20) for j in range(20) if i != j]
np.random.shuffle(pairs)
D1 = torch.stack([Z[i] - Z[j] for i, j in pairs[:32]])

n_sdi, corr_noise = test_noise_stability(D1)
sdi_counts, jaccard_mean = test_reproducibility()
n_survived, corr_drop = test_drop_stability(D1)

print("SDI Stability under noise:")
print(f"  Found: {n_sdi}, Corr: {round(corr_noise, 4)}")
print("\nSDI Reproducibility across trials:")
print(f"  Counts: {sdi_counts}, Jaccard mean: {round(jaccard_mean, 4)}")
print("\nSDI Survival after drop:")
print(f"  Survived: {n_survived}, Corr: {round(corr_drop, 4)}")
