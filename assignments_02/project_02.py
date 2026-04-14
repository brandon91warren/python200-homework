import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# The file uses semicolons as separators, so pd.read_csv() needs sep=";"

os.makedirs("outputs", exist_ok=True)

# Task 1
print("\n--- Task 1 ---")

df = pd.read_csv("student_performance_math.csv", sep=";")

print("Shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

print("\nSaved: outputs/g3_distribution.png")


# Task 2
print("\n--- Task 2 ---")

df_clean = df[df["G3"] > 0].copy()

print("Original shape:", df.shape)
print("Filtered shape:", df_clean.shape)
print("Rows removed:", df.shape[0] - df_clean.shape[0])

# Removing G3=0 rows makes sense because those zeros represent missed final exams,
# not true academic performance. Keeping them would distort the model by mixing
# absence behavior with actual grade outcomes.

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df[col] = df[col].map({"no": 0, "yes": 1})
    df_clean[col] = df_clean[col].map({"no": 0, "yes": 1})

df["sex"] = df["sex"].map({"F": 0, "M": 1})
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Correlation between absences and G3 (original):", corr_original)
print("Correlation between absences and G3 (filtered):", corr_filtered)

# In the original data, students with G3=0 often had very high absences because
# many missed the final exam entirely. That makes absences look less like a clean
# academic-performance predictor and more like a mixture of disengagement and
# non-participation. After filtering, the relationship better reflects students
# who actually completed the course outcome.


# Task 3
print("\n--- Task 3 ---")

numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc",
    "schoolsup", "internet", "higher", "activities", "sex"
]

correlations = df_clean[numeric_features + ["G3"]].corr()["G3"].drop("G3").sort_values()

print("Correlations with G3 (sorted):")
print(correlations)

strongest_feature = correlations.abs().idxmax()
print("\nStrongest relationship with G3:", strongest_feature, correlations[strongest_feature])

plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("G3 vs Failures")
plt.xlabel("Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_vs_failures.png")
plt.close()

print("Saved: outputs/g3_vs_failures.png")

# This plot usually shows a clear downward pattern: students with more past failures
# tend to have lower final math grades.

plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("G3 vs Study Time")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_vs_studytime.png")
plt.close()

print("Saved: outputs/g3_vs_studytime.png")

# This plot helps show whether more study time is associated with higher grades.
# The relationship may be positive, but it is usually noisier than failures.

plt.figure()
plt.scatter(df_clean["absences"], df_clean["G3"])
plt.title("G3 vs Absences")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_vs_absences.png")
plt.close()

print("Saved: outputs/g3_vs_absences.png")

# This plot helps check whether frequent absences are linked to lower performance.
# Even if the trend is negative, the spread is often wide.


# Task 4
print("\n--- Task 4 ---")

X_base = df_clean[["failures"]].values
y_base = df_clean["G3"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_train_base, y_train_base)

y_pred_base = model_base.predict(X_test_base)
rmse_base = np.sqrt(np.mean((y_pred_base - y_test_base) ** 2))
r2_base = model_base.score(X_test_base, y_test_base)

print("Slope:", model_base.coef_[0])
print("RMSE:", rmse_base)
print("R^2 on test set:", r2_base)

# The slope tells us how much the predicted final grade changes for each additional
# past failure. A negative slope means more failures are associated with lower G3.
# RMSE shows the typical prediction error in grade points on a 0-20 scale, so it
# gives a practical sense of how far off predictions usually are.
# R^2 shows how much of the variation in grades is explained by failures alone.


# Task 5
print("\n--- Task 5 ---")

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)
y_pred = model_full.predict(X_test)
rmse_full = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("Test RMSE:", rmse_full)
print("Baseline Test R^2:", r2_base)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Comparing the baseline to the full model shows how much predictive value is added
# by the broader student background and behavior features.
# If train R^2 and test R^2 are close, the model is generalizing reasonably well.
# If train R^2 is much higher, that suggests overfitting.

coef_series = pd.Series(model_full.coef_, index=feature_cols).sort_values()
print("\nCoefficients sorted:")
print(coef_series)

# Any surprising sign may happen because features overlap with each other.
# For example, a variable that looks positive alone could become smaller or even
# negative once the model controls for other related variables.
# In production, I would usually keep the strongest and most stable features and
# consider dropping weak or noisy ones with tiny coefficients or unclear value.


# Task 6
print("\n--- Task 6 ---")

plt.figure()
plt.scatter(y_pred, y_test)

min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("Saved: outputs/predicted_vs_actual.png")

# A point above the diagonal means the actual grade is higher than predicted.
# A point below the diagonal means the actual grade is lower than predicted.
# If the spread grows at the high or low end, that suggests the model struggles more
# in that range. If the spread looks fairly even, error is more uniform.

print("\nSummary:")
print("Filtered dataset size:", df_clean.shape)
print("Test set size:", X_test.shape)
print("Best model RMSE:", rmse_full)
print("Best model Test R^2:", test_r2)

coef_sorted = pd.Series(model_full.coef_, index=feature_cols).sort_values()
largest_negative = coef_sorted.index[0]
second_negative = coef_sorted.index[1]
largest_positive = coef_sorted.index[-1]
second_positive = coef_sorted.index[-2]

print("Largest negative coefficient:", largest_negative, coef_sorted[largest_negative])
print("Second largest negative coefficient:", second_negative, coef_sorted[second_negative])
print("Largest positive coefficient:", largest_positive, coef_sorted[largest_positive])
print("Second largest positive coefficient:", second_positive, coef_sorted[second_positive])

# In plain language, RMSE tells us the model is typically off by about that many
# grade points out of 20. R^2 tells us how much of the grade variation the model
# can explain using these features.
# The largest positive coefficients mean those features are associated with higher
# predicted final grades, holding other variables constant.
# The largest negative coefficients mean those features are associated with lower
# predicted final grades, holding other variables constant.


print("\n--- Neglected Feature: The Power of G1 ---")

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)

print("Test R^2 with G1 added:", test_r2_g1)

# A high R^2 with G1 included does not mean G1 causes G3. It means G1 is a very
# strong earlier indicator of performance in the same course.
# This can be useful for identifying students who may struggle once first-period
# grades are available, but it is less useful for very early intervention.
# If educators want to intervene before G1 exists, they need models based on
# background, behavior, attendance, and other earlier signals instead.