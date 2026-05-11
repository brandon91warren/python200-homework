import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Preprocessing Q1")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit on X_train only so information from the test set does not leak into training.
print("Preprocessing Q2")
print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))
print()

# --- KNN ---
# Q1
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)

print("KNN Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_unscaled))
print("Classification Report:")
print(classification_report(y_test, y_pred_unscaled))
print()

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("KNN Q2")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
# Scaling may make little difference here because the Iris features are already on fairly similar ranges.
print()

# Q3
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

print("KNN Q3")
print("Fold scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Std CV score:", cv_scores.std())
# Cross-validation is more trustworthy than a single train/test split because it averages results across multiple folds.
print()

# Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("KNN Q4")
best_k = None
best_score = -1

for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k}, mean CV score={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("Chosen k:", best_k)
# I would choose the k with the highest mean CV score because it performed best across the validation folds.
print()

# --- Classifier Evaluation ---
# Q1
cm_knn = confusion_matrix(y_test, y_pred_unscaled)
print("Classifier Evaluation Q1")
print("Confusion Matrix:")
print(cm_knn)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png", bbox_inches="tight")
plt.show()

# The model most often confuses versicolor and virginica, if any confusion appears.
print()

# --- Decision Trees ---
# Q1
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)

print("Decision Trees Q1")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print("Classification Report:")
print(classification_report(y_test, tree_pred))
# Compare this accuracy to KNN by checking which printed score is higher.
# Scaled vs. unscaled data usually does not affect a Decision Tree because trees split on feature thresholds, not distances.
print()

# --- Logistic Regression and Regularization ---
# Q1
print("Logistic Regression Q1")
for c_value in [0.01, 1.0, 100]:
    model = LogisticRegression(C=c_value, max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    coef_size = np.abs(model.coef_).sum()
    print(f"C={c_value}, total coefficient magnitude={coef_size:.6f}")

# As C increases, the total coefficient magnitude usually increases.
# This shows that weaker regularization allows the model to use larger coefficients.
print()

# --- PCA ---
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Q1
print("PCA Q1")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)
print()

fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png", bbox_inches="tight")
plt.show()

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits")
plt.colorbar(scatter, label='Digit')
plt.savefig("outputs/pca_2d_projection.png", bbox_inches="tight")
plt.show()

# Same-digit images do tend to cluster together somewhat in this 2D space, though there is still overlap.
print("PCA Q2 completed")
print()

# Q3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("outputs/pca_variance_explained.png", bbox_inches="tight")
plt.show()

components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print("PCA Q3")
print("Components needed for about 80% variance:", components_80)
# It takes about this many components to explain 80% of the variance.
print()

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

sample_indices = [0, 1, 2, 3, 4]
n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, len(sample_indices), figsize=(10, 10))

for col, idx in enumerate(sample_indices):
    axes[0, col].imshow(images[idx], cmap='gray_r')
    axes[0, col].set_title(f"Orig {y_digits[idx]}")
    axes[0, col].axis("off")

for row, n in enumerate(n_values, start=1):
    for col, idx in enumerate(sample_indices):
        reconstructed = reconstruct_digit(idx, scores, pca, n)
        axes[row, col].imshow(reconstructed, cmap='gray_r')
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches="tight")
plt.show()

# The digits usually become clearly recognizable around 15 components, and that generally matches where the variance curve starts leveling off.
print("PCA Q4 completed")
print()