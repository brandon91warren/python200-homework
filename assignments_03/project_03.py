# project_03.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)

# --- Load data ---
columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "spam_label"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam_df = pd.read_csv(url, header=None, names=columns)

X = spam_df.drop(columns="spam_label")
y = spam_df["spam_label"]

# --- Task 1: Load and Explore ---
print("Task 1")
print("Dataset shape:", spam_df.shape)
print("Number of emails:", len(spam_df))
print("Class counts:")
print(y.value_counts())
print("Class proportions:")
print(y.value_counts(normalize=True))
print()

# The classes are somewhat imbalanced, with more ham than spam, so accuracy alone can be misleading.
# A model could get a decent accuracy score just by favoring the majority ham class.

for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    plt.figure(figsize=(6, 4))
    spam_df.boxplot(column=feature, by="spam_label")
    plt.title(f"{feature} by spam_label")
    plt.suptitle("")
    plt.xlabel("spam_label")
    plt.ylabel(feature)
    safe_name = feature.replace("!", "exclam").replace("$", "dollar").replace(";", "semicolon")
    plt.savefig(f"outputs/{safe_name}_boxplot.png", bbox_inches="tight")
    plt.show()

print("Selected feature summary by class:")
for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    print(f"\n{feature}")
    print(spam_df.groupby("spam_label")[feature].describe())

# The differences between spam and ham are fairly noticeable for these features.
# Spam emails tend to have higher values for word_freq_free, char_freq_!, and capital_run_length_total.

# The heavy skew toward zero means many word-frequency features are sparse because most emails do not contain most tracked words.
# The numeric scales vary a lot because some features are small frequencies while others are capital-run statistics that can be much larger.
# This matters for KNN, logistic regression, and PCA because larger-scale features can dominate unless the data is scaled.

print("\nOverall feature ranges:")
print(X.describe().T[["mean", "std", "min", "max"]].head(15))
print()

# --- Task 2: Prepare Your Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# I used a train/test split so I could evaluate each model on unseen data.
# I scaled the data because KNN, logistic regression, and PCA are sensitive to feature magnitude.

# I fit the scaler on X_train only so the test set does not leak information into preprocessing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# I fit PCA on the scaled training data only because PCA is also a learned preprocessing step and should not use test-set information.
# I kept both the full scaled data and the PCA-reduced data so I could compare whether dimensionality reduction helps the non-tree models.
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(0.90, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Spambase PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("outputs/spambase_pca_variance.png", bbox_inches="tight")
plt.show()

n = np.argmax(cumulative_variance >= 0.90) + 1
print("Task 2")
print("Components to reach 90% explained variance:", n)
print()

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# --- Task 3: Classifier Comparison ---

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(name)
    print("Accuracy:", acc)
    print(classification_report(yte, preds))
    print()
    return model, preds, acc

results = {}

knn_unscaled_model, knn_unscaled_preds, knn_unscaled_acc = evaluate_model(
    "KNN unscaled",
    KNeighborsClassifier(n_neighbors=5),
    X_train, X_test, y_train, y_test
)
results["KNN unscaled"] = knn_unscaled_acc

knn_scaled_model, knn_scaled_preds, knn_scaled_acc = evaluate_model(
    "KNN scaled",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, X_test_scaled, y_train, y_test
)
results["KNN scaled"] = knn_scaled_acc

knn_pca_model, knn_pca_preds, knn_pca_acc = evaluate_model(
    "KNN PCA",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca, X_test_pca, y_train, y_test
)
results["KNN PCA"] = knn_pca_acc

# Scaling improved KNN dramatically because KNN depends on distance and is very sensitive to feature scale.
# PCA did not improve KNN on the test split here, although it stayed very close.

print("Decision Tree depth search")
tree_depth_results = {}
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    tree_depth_results[depth] = test_acc
    print(f"max_depth={depth}, train_accuracy={train_acc:.4f}, test_accuracy={test_acc:.4f}")
print()

# As depth increases, training accuracy rises much faster than test accuracy.
# This shows overfitting, especially at max_depth=None where the tree nearly memorizes the training data.

# I would use max_depth=10 in production because it gives strong test accuracy while avoiding the extreme overfitting shown by an unlimited-depth tree.
best_tree_depth = 10

tree_model, tree_preds, tree_acc = evaluate_model(
    f"Decision Tree (max_depth={best_tree_depth})",
    DecisionTreeClassifier(max_depth=best_tree_depth, random_state=42),
    X_train, X_test, y_train, y_test
)
results["Decision Tree"] = tree_acc

rf_model, rf_preds, rf_acc = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)
results["Random Forest"] = rf_acc

log_scaled_model, log_scaled_preds, log_scaled_acc = evaluate_model(
    "Logistic Regression scaled",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled, X_test_scaled, y_train, y_test
)
results["Logistic Regression scaled"] = log_scaled_acc

log_pca_model, log_pca_preds, log_pca_acc = evaluate_model(
    "Logistic Regression PCA",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca, X_test_pca, y_train, y_test
)
results["Logistic Regression PCA"] = log_pca_acc

# Logistic regression performed better on the scaled data than on the PCA-reduced data.
# That suggests PCA removed some information that was still useful for classification.

print("Task 3 Summary")
for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.4f}")
print()

# Random Forest performed best overall on this dataset.
# Among the models where I compared PCA vs. non-PCA, PCA did not improve logistic regression and was only roughly comparable for KNN.
# That mostly matches the idea from Task 2 that scaling is essential, while PCA may or may not help depending on how much useful information is lost.

# For a spam filter, accuracy is helpful but not the only metric that matters.
# I would prioritize reducing false positives because marking a legitimate email as spam can be more harmful than letting one spam email get through.

best_model_name = max(results, key=results.get)
print("Best model:", best_model_name)

if best_model_name == "KNN unscaled":
    best_preds = knn_unscaled_preds
elif best_model_name == "KNN scaled":
    best_preds = knn_scaled_preds
elif best_model_name == "KNN PCA":
    best_preds = knn_pca_preds
elif best_model_name == "Decision Tree":
    best_preds = tree_preds
elif best_model_name == "Random Forest":
    best_preds = rf_preds
elif best_model_name == "Logistic Regression scaled":
    best_preds = log_scaled_preds
else:
    best_preds = log_pca_preds

best_cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["ham", "spam"])
disp.plot()
plt.title(f"Best Model Confusion Matrix: {best_model_name}")
plt.savefig("outputs/best_model_confusion_matrix.png", bbox_inches="tight")
plt.show()

tn, fp, fn, tp = best_cm.ravel()
print("Best model confusion matrix:")
print(best_cm)
print("False positives:", fp)
print("False negatives:", fn)
print()

# This best model makes more false negatives than false positives.
# In other words, it lets more spam slip through than it incorrectly marks legitimate email as spam.

# --- Feature Importances ---
tree_for_importance = DecisionTreeClassifier(max_depth=best_tree_depth, random_state=42)
tree_for_importance.fit(X_train, y_train)

rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_importance.fit(X_train, y_train)

tree_importances = pd.Series(tree_for_importance.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importances = pd.Series(rf_for_importance.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Top 10 Decision Tree feature importances")
print(tree_importances.head(10))
print()

print("Top 10 Random Forest feature importances")
print(rf_importances.head(10))
print()

# The Decision Tree and Random Forest agree on several important features, especially punctuation and money-related signals.
# Features like char_freq_!, char_freq_$, word_freq_remove, and word_freq_free match common patterns people associate with spam.

plt.figure(figsize=(10, 6))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("outputs/feature_importances.png", bbox_inches="tight")
plt.show()

# --- Task 4: Cross-Validation ---
print("Task 4 Cross-Validation")

cv_results = {}

cv_models = {
    "KNN unscaled": (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN scaled": (Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]), X_train),
    "KNN PCA": (Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]), X_train),
    "Decision Tree": (DecisionTreeClassifier(max_depth=best_tree_depth, random_state=42), X_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression scaled": (Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ]), X_train),
    "Logistic Regression PCA": (Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ]), X_train),
}

for name, (model, X_cv) in cv_models.items():
    scores = cross_val_score(model, X_cv, y_train, cv=5)
    cv_results[name] = (scores.mean(), scores.std())
    print(name)
    print("Mean CV accuracy:", scores.mean())
    print("Std CV accuracy:", scores.std())
    print()

print("CV ranking")
for name, (mean_score, std_score) in sorted(cv_results.items(), key=lambda x: x[1][0], reverse=True):
    print(f"{name}: mean={mean_score:.4f}, std={std_score:.4f}")
print()

# Random Forest is the most accurate model across cross-validation.
# Logistic Regression PCA is the most stable by standard deviation, with Logistic Regression scaled essentially tied.
# The ranking is mostly consistent with the single train/test split, which increases confidence in the results.

# --- Task 5: Prediction Pipelines ---

best_tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

best_non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

if log_pca_acc > log_scaled_acc:
    best_non_tree_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ])

best_tree_pipeline.fit(X_train, y_train)
tree_pipe_preds = best_tree_pipeline.predict(X_test)

print("Task 5 Best Tree Pipeline")
print(classification_report(y_test, tree_pipe_preds))
print()

best_non_tree_pipeline.fit(X_train, y_train)
non_tree_pipe_preds = best_non_tree_pipeline.predict(X_test)

print("Task 5 Best Non-Tree Pipeline")
print(classification_report(y_test, non_tree_pipe_preds))
print()

# The two pipelines do not have the same structure because tree-based models do not require scaling or PCA.
# The non-tree pipeline includes preprocessing steps because those models are sensitive to feature magnitude and, if used, dimensionality reduction.

# Packaging the model in a pipeline makes the workflow easier to reuse, reduces the chance of forgetting a preprocessing step,
# and makes the model easier to hand off to someone else or deploy in a consistent way.