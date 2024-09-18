import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Charger les données depuis un fichier CSV
file_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\donnees_recensement_nettoyees.csv'
data = pd.read_csv(file_path)

# Vérifiez les premières lignes pour vous assurer que les colonnes nécessaires sont présentes
print(data.head())

# 2. Normalisation des données (DBSCAN est sensible à la magnitude des variables)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['age', 'revenu']])  # Nous ne normalisons que les colonnes 'age' et 'revenu'

# 3. Clustering avec DBSCAN (ajuster les paramètres eps et min_samples)
dbscan = DBSCAN(eps=0.12, min_samples=10)  # Essayez de réduire eps et augmenter min_samples
data['cluster'] = dbscan.fit_predict(data_scaled)

# Vérifiez l'ajout de la colonne 'cluster'
print(data.head())

# 4. Fonction pour visualiser les clusters avec Seaborn
def visualiser_clusters(data):
    # Créer un pairplot pour visualiser les relations entre 'age', 'revenu' et les clusters
    sns.pairplot(data, hue="cluster", vars=["age", "revenu"], palette="husl", diag_kind="kde")
    
    # Afficher le graphique
    plt.show()

# 5. Appel de la fonction pour visualiser les clusters
visualiser_clusters(data)
