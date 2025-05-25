import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configuration
num_classes = 4                     # Number of target classes
n_samples = 1000                    # Total samples
noise_levels = np.linspace(0, 1, 25)  # Range of label noise (η)
accuracies = []                     # Store accuracies for each η

# Generate synthetic classification dataset
X, y_true = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=num_classes,
    n_clusters_per_class=1,
    random_state=42
)

# Split into train and test
X_train, X_test, y_train_clean, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# Simulate classification under increasing label noise
for eta in noise_levels:
    y_train_noisy = y_train_clean.copy()
    
    # Inject noise: flip labels at η proportion of training set
    num_noisy = int(eta * len(y_train_noisy))
    noisy_indices = np.random.choice(len(y_train_noisy), size=num_noisy, replace=False)
    y_train_noisy[noisy_indices] = np.random.randint(0, num_classes, size=num_noisy)

    # Train classifier on noisy data
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    clf.fit(X_train, y_train_noisy)

    # Evaluate on clean test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', label='Empirical Accuracy', color='blue')
plt.axhline(1/num_classes, color='red', linestyle='--', label='Random Guessing (1/C)')
plt.xlabel('Noise Level (η)')
plt.ylabel('Classification Accuracy')
plt.title('Adaptation Breakdown under Increasing Label Noise')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
