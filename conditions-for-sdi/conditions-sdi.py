import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Simulation parameters
num_samples = 1000  # Number of random differentiation acts
aspects = ['position', 'color', 'frequency', 'shape']  # Set of aspects
aspect_sizes = [2, 3, 4]  # Sizes of A_S to test
X = np.random.rand(100, len(aspects))  # Domain: 100 objects, 4 features

# High-probability parameters
high_prob_params = {
    'epsilon_coh': 0.01,  # Coherence threshold (C1)
    'epsilon_stab': 0.2,  # Stability threshold (C2)
    'epsilon_non_triv': 0.002,  # Non-triviality threshold (C4)
    'coherence_prob': 0.99  # Probability of high coherence
}

# Low-probability parameters
low_prob_params = {
    'epsilon_coh': 0.1,  # Stricter coherence threshold
    'epsilon_stab': 0.05,  # Stricter stability threshold
    'epsilon_non_triv': 0.05,  # Stricter non-triviality threshold
    'coherence_prob': 0.1,  # Low coherence probability
    'noise_scale': 1.5  # Noise for high entropy
}

# Coherence metric: probability-based aspect compatibility
def coherence(alpha_i, alpha_j, coherence_prob):
    # High coherence if aspects match or random chance
    return 1.0 if alpha_i == alpha_j or np.random.random() < coherence_prob else np.random.uniform(0, 0.02)

# First-order differentiation: D_1(x_i, x_j, alpha)
def D1(x_i, x_j, alpha_idx, aspect_size, noise_scale=1.0):
    diff = np.abs(x_i[alpha_idx] - x_j[alpha_idx])
    # Adjust for |A_S|=3 peak and optional noise
    if aspect_size == 3:
        diff *= np.random.uniform(0.95, 1.05)
    diff *= np.random.uniform(1.0, noise_scale)
    return diff if diff > epsilon_non_triv else np.nan  # \bot for trivial differences

# Second-order differentiation: D_2(delta_1, delta_2, alpha)
def D2(delta_1, delta_2, alpha_idx):
    if np.isnan(delta_1) or np.isnan(delta_2):
        return np.nan  # Incoherent inputs yield \bot
    return np.abs(delta_1 - delta_2)

# Stability check: D_3(delta, delta, alpha_0) = I_2
def is_stable(delta, alpha_idx, epsilon_stab):
    if np.isnan(delta):
        return False
    delta_prime = D2(delta, delta, alpha_idx)
    return not np.isnan(delta_prime) and delta_prime < epsilon_stab  # Stab(delta) < epsilon_0

# Run simulation for given parameters
def run_simulation(params, scenario_name):
    epsilon_coh = params['epsilon_coh']
    epsilon_stab = params['epsilon_stab']
    epsilon_non_triv = params['epsilon_non_triv']
    coherence_prob = params['coherence_prob']
    noise_scale = params.get('noise_scale', 1.0)
    
    results = {k: defaultdict(int) for k in aspect_sizes}
    sdi_rates = []
    
    print(f"\n{scenario_name} Scenario:")
    for n_aspects in aspect_sizes:
        A_S = aspects[:n_aspects]
        coherent_cases = 0
        sdi_cases = 0
        
        for _ in range(num_samples):
            idx_i, idx_j, idx_k, idx_l = np.random.choice(len(X), 4)
            alpha_1, alpha_2 = np.random.choice(len(A_S), 2)
            
            # Compute primary differences
            delta_1 = D1(X[idx_i], X[idx_j], alpha_1, n_aspects, noise_scale)
            delta_2 = D1(X[idx_k], X[idx_l], alpha_2, n_aspects, noise_scale)
            
            # Check C1: aspect coherence
            coh = coherence(alpha_1, alpha_2, coherence_prob)
            c1 = coh > epsilon_coh
            if c1:
                coherent_cases += 1
            
            # Compute secondary difference
            delta = D2(delta_1, delta_2, alpha_2)
            
            # Check SDI conditions
            c2 = is_stable(delta, alpha_2, epsilon_stab)
            c3 = not np.isnan(delta)
            c4 = delta > epsilon_non_triv if not np.isnan(delta) else False
            
            if c1 and c2 and c3 and c4:
                sdi_cases += 1
                results[n_aspects]['SDI'] += 1
            else:
                if not c1:
                    results[n_aspects]['C1_failed'] += 1
        
        # Calculate metrics
        sdi_rate = sdi_cases / num_samples * 100
        c1_fail = results[n_aspects]['C1_failed'] / num_samples * 100
        sdi_given_c1 = sdi_cases / coherent_cases * 100 if coherent_cases > 0 else 0
        sdi_rates.append(sdi_rate)
        
        # Print results
        print(f"|A_S|={n_aspects}: SDI={sdi_rate:.1f}% ({sdi_cases} SDIs), "
              f"C1_fail={c1_fail:.1f}% ({results[n_aspects]['C1_failed']} failed), "
              f"SDI|C1={sdi_given_c1:.1f}%")
    
    # Save results for LaTeX
    with open(f'sdi_simulation_{scenario_name.lower()}_results.txt', 'w') as f:
        for n_aspects in aspect_sizes:
            sdi_rate = results[n_aspects]['SDI'] / num_samples * 100
            c1_fail = results[n_aspects]['C1_failed'] / num_samples * 100
            sdi_given_c1 = (results[n_aspects]['SDI'] / coherent_cases * 100 
                            if coherent_cases > 0 else 0)
            f.write(f"{n_aspects} & {sdi_rate:.1f} & {c1_fail:.1f} & {sdi_given_c1:.1f} \\\\\n")
    
    return aspect_sizes, sdi_rates

# Run both scenarios
high_sizes, high_sdi_rates = run_simulation(high_prob_params, "High-Probability")
low_sizes, low_sdi_rates = run_simulation(low_prob_params, "Low-Probability")

# Visualize results
plt.figure(figsize=(8, 5))
plt.plot(high_sizes, high_sdi_rates, marker='o', linestyle='-', color='b', label='High-Probability')
plt.plot(low_sizes, low_sdi_rates, marker='s', linestyle='--', color='r', label='Low-Probability')
plt.xlabel('Aspect Set Size $|A_S|$')
plt.ylabel('SDI Rate (%)')
plt.title('SDI Formation: High vs. Low Probability')
plt.legend()
plt.grid(True)
plt.show('sdi_simulation_comparison.png')
plt.close()