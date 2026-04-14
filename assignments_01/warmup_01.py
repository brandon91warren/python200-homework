import pandas as pd

# --- Pandas ---

# Pandas Q1
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

print("# Pandas Q1")
print("First 3 rows:")
print(df.head(3))
print(f"Shape: {df.shape}")
print("Data types:")
print(df.dtypes)
print()

# Pandas Q2
passed_above_80 = df[(df["passed"] == True) & (df["grade"] > 80)]

print("# Pandas Q2")
print("Students who passed and have a grade above 80:")
print(passed_above_80)
print()

# Pandas Q3
df["grade_curved"] = df["grade"] + 5

print("# Pandas Q3")
print("Updated DataFrame with grade_curved:")
print(df)
print()

# Pandas Q4
df["name_upper"] = df["name"].str.upper()

print("# Pandas Q4")
print("Name and name_upper columns:")
print(df[["name", "name_upper"]])
print()

# Pandas Q5
mean_grade_by_city = df.groupby("city")["grade"].mean()

print("# Pandas Q5")
print("Mean grade by city:")
print(mean_grade_by_city)
print()

# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")

print("# Pandas Q6")
print("Name and city columns after replacing Austin with Houston:")
print(df[["name", "city"]])
print()

# Pandas Q7
sorted_df = df.sort_values(by="grade", ascending=False)

print("# Pandas Q7")
print("Top 3 rows sorted by grade descending:")
print(sorted_df.head(3))
print()

import numpy as np

# --- NumPy ---

# NumPy Q1
arr1 = np.array([10, 20, 30, 40, 50])

print("# NumPy Q1")
print("Array:", arr1)
print(f"Shape: {arr1.shape}")
print(f"Dtype: {arr1.dtype}")
print(f"Number of dimensions: {arr1.ndim}")
print()

# NumPy Q2
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print("# NumPy Q2")
print("Array:")
print(arr2)
print(f"Shape: {arr2.shape}")
print(f"Size: {arr2.size}")
print()

# NumPy Q3
slice_2x2 = arr2[:2, :2]

print("# NumPy Q3")
print("Top-left 2x2 block:")
print(slice_2x2)
print()

# NumPy Q4
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))

print("# NumPy Q4")
print("3x4 zeros array:")
print(zeros_array)
print("2x5 ones array:")
print(ones_array)
print()

# NumPy Q5
arr_range = np.arange(0, 50, 5)

print("# NumPy Q5")
print("Array from np.arange(0, 50, 5):")
print(arr_range)
print(f"Shape: {arr_range.shape}")
print(f"Mean: {arr_range.mean()}")
print(f"Sum: {arr_range.sum()}")
print(f"Standard Deviation: {arr_range.std()}")
print()

# NumPy Q6
random_arr = np.random.normal(0, 1, 200)

print("# NumPy Q6")
print(f"Mean of random array: {random_arr.mean()}")
print(f"Standard Deviation of random array: {random_arr.std()}")
print()

import matplotlib.pyplot as plt

# --- Matplotlib ---

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.legend(["Dataset 1", "Dataset 2"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q4
fig, axs = plt.subplots(1, 2)

# Left plot (line)
axs[0].plot(x, y)
axs[0].set_title("Squares")

# Right plot (bar)
axs[1].bar(subjects, scores)
axs[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()
