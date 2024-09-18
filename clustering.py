import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clustering_analyse(file_path):
    # Charger les données nettoyées
    data = pd.read_csv(file_path)

    # Préparation des données pour le clustering
    # Encodage des variables catégorielles
    data_encoded = data.copy()
    data_encoded['sexe'] = data_encoded['sexe'].map({'M': 1, 'F': 0})  # Encodage de sexe en numérique
    data_encoded = pd.get_dummies(data_encoded, columns=['categorie_professionnelle'], drop_first=True)

    # Normalisation des données pour KMeans et DBSCAN
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    # 1. Clustering avec K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)  # Vous pouvez changer le nombre de clusters
    kmeans_labels = kmeans.fit_predict(data_scaled)

    # 2. Clustering avec DBSCAN avec paramètres modifiés
    dbscan = DBSCAN(eps=0.7, min_samples=10)  # Augmentation de eps et de min_samples
    dbscan_labels = dbscan.fit_predict(data_scaled)

    # 3. Réduction de dimension pour visualisation avec PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Ajouter les labels des clusters dans les données originales
    data['kmeans_cluster'] = kmeans_labels
    data['dbscan_cluster'] = dbscan_labels

    # 4. Visualisation des clusters
    # Visualisation des clusters K-Means
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=kmeans_labels, palette='viridis', s=100)
    plt.title('Clusters K-Means')
    plt.show()

    # Visualisation des clusters DBSCAN
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=dbscan_labels, palette='coolwarm', s=100)
    plt.title('Clusters DBSCAN')
    plt.show()

    # 5. Comparaison des clusters
    print("\nComparaison des clusters K-Means :")
    print(data['kmeans_cluster'].value_counts())

    print("\nComparaison des clusters DBSCAN :")
    print(data['dbscan_cluster'].value_counts())

# Utilisation de la fonction
file_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\donnees_recensement_nettoyees.csv'
clustering_analyse(file_path)
