
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


# =====================================================
# S1: LOAD THE DATASET
# =====================================================

df = pd.read_csv("Mall_Customers.csv")

print("FIRST FIVE ROWS OF THE DATASET:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# =====================================================
# S2: DATA PREPROCESSING
# =====================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates().copy()

# Handle missing values if present
# Numerical columns: replace missing values with the median
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: replace missing values with the mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nDataset shape after preprocessing:", df.shape)
print("Remaining missing values:", df.isnull().sum().sum())


# =====================================================
# S3: FEATURE SELECTION
# =====================================================

# CustomerID is an identifier, so exclude it.
# Gender is excluded from this numerical K-Means model.
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\nSelected features:")
print(X.head())

print("\nFeature names:", X.columns.tolist())
print("Selected data shape:", X.shape)


# =====================================================
# S4: FEATURE SCALING AND K-MEANS
# =====================================================

# Scale the features because they have different ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------------------------------
# Elbow Method and Silhouette Score comparison
# -----------------------------------------------------

inertia = []
silhouette_scores = []
k_values = range(2, 11)

print("\nCOMPARING DIFFERENT NUMBERS OF CLUSTERS:")

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Inertia = {model.inertia_:.2f}, "
        f"Silhouette Score = {score:.3f}"
    )


# Choose K based on the elbow plot and score comparison.
# K=5 is the initial choice; change it after evaluating results.
optimal_k = 5

print("\nSelected number of clusters:", optimal_k)

# Fit the final K-Means model
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

final_score = silhouette_score(X_scaled, df["Cluster"])

print("Final Silhouette Score:", round(final_score, 3))

print("\nCustomer count in each cluster:")
print(df["Cluster"].value_counts().sort_index())


# =====================================================
# S5: VISUALIZATION AND OUTPUT
# =====================================================

# -----------------------------------------------------
# 1. Elbow Method
# -----------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 2. Silhouette Score Comparison
# -----------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 3. t-SNE Visualization
# -----------------------------------------------------

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.8
)

plt.title("Customer Segmentation Using K-Means and t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 4. Customer Distribution
# -----------------------------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind="bar")

plt.title("Customer Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 5. Analyze Cluster Characteristics
# -----------------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean().round(2)

print("\nAVERAGE CHARACTERISTICS OF EACH CLUSTER:")
print(cluster_summary)

print("\nCLUSTER SIZES:")
print(cluster_counts)


# -----------------------------------------------------
# 6. Display Final Customer Assignments
# -----------------------------------------------------

print("\nFINAL CUSTOMER SEGMENTATION:")
print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"
        ]
    ].head(10)
)


# -----------------------------------------------------
# 7. Save Results
# -----------------------------------------------------

df.to_csv("customer_segments.csv", index=False)

cluster_summary.to_csv("cluster_summary.csv")

print("\nFiles saved successfully:")
print("1. customer_segments.csv")
print("2. cluster_summary.csv")

print("\nCUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
