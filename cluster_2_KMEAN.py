import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Charger les données depuis un fichier CSV
file_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\donnees_recensement_nettoyees.csv'
data = pd.read_csv(file_path)

# Vérifiez les premières lignes pour vous assurer que les colonnes nécessaires sont présentes
print(data.head())

# 2. Vérifiez que les colonnes 'age' et 'revenu' existent
# Si les données sont déjà bien formatées avec 'age' et 'revenu', on peut passer directement au clustering.
# Si vous devez faire un clustering, ajoutez les résultats à la DataFrame :
kmeans = KMeans(n_clusters=4, random_state=42)  # Vous pouvez ajuster le nombre de clusters
data['cluster'] = kmeans.fit_predict(data[['age', 'revenu']])  # Utilisez les colonnes 'age' et 'revenu' pour calculer les clusters

# Vérifiez l'ajout de la colonne 'cluster'
print(data.head())

# 3. Fonction pour visualiser les clusters avec Seaborn
def visualiser_clusters(data):
    # Créer un pairplot pour visualiser les relations entre 'age', 'revenu' et les clusters
    sns.pairplot(data, hue="cluster", vars=["age", "revenu"], palette="husl", diag_kind="kde")
    
    # Afficher le graphique
    plt.show()

# 4. Appel de la fonction pour visualiser les clusters
visualiser_clusters(data)
