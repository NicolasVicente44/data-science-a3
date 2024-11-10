import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os



base_path = os.path.dirname(__file__)
heart_path = os.path.join(base_path, "Heart.csv")
credit_path = os.path.join(base_path, "Credit.csv")

# hheart dataset

heart_data = pd.read_csv(heart_path).drop(columns=["Unnamed: 0"])
heart_data = pd.get_dummies(heart_data, columns=["ChestPain", "Thal"])
imputer = SimpleImputer(strategy="mean")
heart_data_imputed = imputer.fit_transform(heart_data.select_dtypes(include=[float, int]))
scaler = StandardScaler()
heart_data_scaled = scaler.fit_transform(heart_data_imputed)

kmeans_heart = KMeans(n_clusters=3, random_state=0).fit(heart_data_scaled)
heart_data["Cluster"] = kmeans_heart.labels_
print("Heart Dataset Cluster Sizes:\n", pd.Series(heart_data["Cluster"]).value_counts())
print("Silhouette Score for Heart Dataset Clusters:", silhouette_score(heart_data_scaled, kmeans_heart.labels_))
print("Descriptive Statistics for Heart Dataset Clusters:\n", heart_data.groupby("Cluster")[["Age", "Chol", "RestBP", "MaxHR"]].agg(["mean", "std", "min", "max"]))

pca_heart = PCA(n_components=2)
heart_data_pca = pca_heart.fit_transform(heart_data_scaled)
heart_centroids_pca = pca_heart.transform(kmeans_heart.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(heart_data_pca[:, 0], heart_data_pca[:, 1], c=kmeans_heart.labels_, cmap="viridis", s=50)
plt.scatter(heart_centroids_pca[:, 0], heart_centroids_pca[:, 1], c="red", s=200, marker="X", label="Centroids")
plt.title("Heart Dataset Clusters with Centroids (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(heart_data["Age"], heart_data["Chol"], c=kmeans_heart.labels_, cmap="viridis", s=50)
centroids_original_space = scaler.inverse_transform(kmeans_heart.cluster_centers_)
plt.scatter(centroids_original_space[:, heart_data.columns.get_loc("Age")], centroids_original_space[:, heart_data.columns.get_loc("Chol")], c="red", s=200, marker="X", label="Centroids")
plt.title("Heart Dataset Clusters with Centroids (Using Age and Cholesterol)")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.colorbar(label="Cluster Label")
plt.legend()
plt.show()


#credit dataset

credit_data = pd.read_csv(credit_path)
credit_data = pd.get_dummies(credit_data, columns=["Own", "Student", "Married", "Region"])
credit_data_imputed = imputer.fit_transform(credit_data.select_dtypes(include=[float, int]))
credit_data_scaled = scaler.fit_transform(credit_data_imputed)

kmeans_credit = KMeans(n_clusters=4, random_state=0).fit(credit_data_scaled)
credit_data["Cluster"] = kmeans_credit.labels_
print("Credit Dataset Cluster Sizes:\n", pd.Series(credit_data["Cluster"]).value_counts())
print("Silhouette Score for Credit Dataset Clusters:", silhouette_score(credit_data_scaled, kmeans_credit.labels_))
print("Descriptive Statistics for Credit Dataset Clusters:\n", credit_data.groupby("Cluster")[["Income", "Limit", "Rating", "Balance"]].agg(["mean", "std", "min", "max"]))

pca_credit = PCA(n_components=2)
credit_data_pca = pca_credit.fit_transform(credit_data_scaled)
credit_centroids_pca = pca_credit.transform(kmeans_credit.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(credit_data_pca[:, 0], credit_data_pca[:, 1], c=kmeans_credit.labels_, cmap="plasma", s=50)
plt.scatter(credit_centroids_pca[:, 0], credit_centroids_pca[:, 1], c="red", s=200, marker="X", label="Centroids")
plt.title("Credit Dataset Clusters with Centroids (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(credit_data["Income"], credit_data["Balance"], c=kmeans_credit.labels_, cmap="plasma", s=50)
centroids_original_space_credit = scaler.inverse_transform(kmeans_credit.cluster_centers_)
plt.scatter(centroids_original_space_credit[:, credit_data.columns.get_loc("Income")], centroids_original_space_credit[:, credit_data.columns.get_loc("Balance")], c="red", s=200, marker="X", label="Centroids")
plt.title("Credit Dataset Clusters with Centroids (Using Income and Balance)")
plt.xlabel("Income")
plt.ylabel("Balance")
plt.colorbar(label="Cluster Label")
plt.legend()
plt.show()


heart_silhouette = silhouette_score(heart_data_scaled, kmeans_heart.labels_)
credit_silhouette = silhouette_score(credit_data_scaled, kmeans_credit.labels_)

print(f"Average Silhouette Score for Heart Dataset: {heart_silhouette:.3f}")
print(f"Average Silhouette Score for Credit Dataset: {credit_silhouette:.3f}")
