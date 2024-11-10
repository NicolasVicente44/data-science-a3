import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(__file__)
heart_path = os.path.join(base_path, "Heart.csv")
credit_path = os.path.join(base_path, "Credit.csv")

heart_data = pd.read_csv(heart_path).drop(columns=["Unnamed: 0"])
heart_data = pd.get_dummies(heart_data, columns=["ChestPain", "Thal"])
imputer = SimpleImputer(strategy="mean")
heart_data_imputed = imputer.fit_transform(heart_data.drop(columns=["AHD"]))
scaler = StandardScaler()
heart_data_scaled = scaler.fit_transform(heart_data_imputed)
kmeans_heart = KMeans(n_clusters=3, random_state=0).fit(heart_data_scaled)
heart_clusters = kmeans_heart.labels_

pca_heart = PCA(n_components=2)
heart_data_pca = pca_heart.fit_transform(heart_data_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(heart_data_pca[:, 0], heart_data_pca[:, 1], c=heart_clusters, cmap="viridis", s=50)
plt.title("Heart Dataset Clusters (Colored by Cluster Label)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")

for i in range(len(heart_data)):
    plt.annotate(int(heart_data.iloc[i]["Age"]), (heart_data_pca[i, 0], heart_data_pca[i, 1]), fontsize=8, alpha=0.7)
plt.show()

credit_data = pd.read_csv(credit_path)
credit_data = pd.get_dummies(credit_data, columns=["Own", "Student", "Married", "Region"])
credit_data_imputed = imputer.fit_transform(credit_data)
credit_data_scaled = scaler.fit_transform(credit_data_imputed)
kmeans_credit = KMeans(n_clusters=4, random_state=0).fit(credit_data_scaled)
credit_clusters = kmeans_credit.labels_

pca_credit = PCA(n_components=2)
credit_data_pca = pca_credit.fit_transform(credit_data_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(credit_data_pca[:, 0], credit_data_pca[:, 1], c=credit_clusters, cmap="plasma", s=50)
plt.title("Credit Dataset Clusters (Colored by Cluster Label)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")

for i in range(len(credit_data)):
    plt.annotate(int(credit_data.iloc[i]["Income"]), (credit_data_pca[i, 0], credit_data_pca[i, 1]), fontsize=8, alpha=0.7)
plt.show()
# Simple 2-feature plot for Heart dataset
plt.figure(figsize=(10, 6))
plt.scatter(heart_data['Age'], heart_data['Chol'], c=heart_clusters, cmap="viridis", s=50)
plt.title("Heart Dataset Clusters (Using Age and Cholesterol)")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.colorbar(label="Cluster Label")
plt.show()
