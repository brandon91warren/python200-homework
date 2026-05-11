import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)

# Q1
print("\n--- Q1 ---")

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction (4 years):", pred_4)
print("Prediction (8 years):", pred_8)


# Q2
print("\n--- Q2 ---")

x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)


# Q3
print("\n--- Q3 ---")

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("Cluster centers:")
print(kmeans.cluster_centers_)

print("Cluster counts:")
print(np.bincount(labels))

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    c="black"
)
plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

print("Saved: outputs/kmeans_clusters.png")


np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# Q4
print("\n--- Q4 ---")

plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.close()

print("Saved: outputs/cost_vs_age.png")

# There appear to be two visible groups in the plot.
# This suggests smoker status has a strong effect on medical cost.


# Q5
print("\n--- Q5 ---")

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q6
print("\n--- Q6 ---")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R^2 on test set:", r2)

# The slope shows how much medical cost is expected to increase
# for each additional year of age.


# Q7
print("\n--- Q7 ---")

X_full = np.column_stack([age, smoker])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("Q6 R^2:", r2)
print("Q7 R^2:", r2_full)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient represents the additional medical cost
# associated with being a smoker, holding age constant.


# Q8
print("\n--- Q8 ---")

y_pred_full = model_full.predict(X_test_full)

plt.figure()
plt.scatter(y_pred_full, y_test_full)

min_val = min(y_pred_full.min(), y_test_full.min())
max_val = max(y_pred_full.max(), y_test_full.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("Saved: outputs/predicted_vs_actual.png")

# A point above the diagonal means the actual value is higher than predicted.
# A point below the diagonal means the actual value is lower than predicted.