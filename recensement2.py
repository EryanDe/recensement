import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyser_visualiser_donnees(file_path):
    # Charger les données nettoyées
    data = pd.read_csv(file_path)

    # 1. Analyse exploratoire des données
    print("Résumé statistique des données :")
    print(data.describe())  # Statistiques descriptives pour les colonnes numériques
    print("\nRésumé des données catégorielles :")
    print(data.select_dtypes(include=['object']).describe())  # Statistiques descriptives pour les colonnes catégorielles

    # 2. Visualisation des données
    # Histogramme de la distribution des âges
    plt.figure(figsize=(10, 6))
    plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution des âges')
    plt.xlabel('Âge')
    plt.ylabel('Nombre de personnes')
    plt.show()

    # Répartition par sexe
    plt.figure(figsize=(6, 6))
    sns.countplot(x='sexe', data=data, palette='Set2')
    plt.title('Répartition par sexe')
    plt.xlabel('Sexe (1 = Homme, 0 = Femme)')
    plt.ylabel('Nombre')
    plt.show()

    # Boxplot du revenu par catégorie professionnelle
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='categorie_professionnelle', y='revenu', data=data, palette='Set3')
    plt.title('Revenu par catégorie professionnelle')
    plt.xlabel('Catégorie professionnelle')
    plt.ylabel('Revenu')
    plt.xticks(rotation=45)
    plt.show()

   # Matrice de corrélation
    # Convertir les colonnes catégorielles en numériques
    data_encoded = data.copy()
    data_encoded['sexe'] = data_encoded['sexe'].map({'M': 1, 'F': 0})  # Convertir sexe en numérique
    data_encoded = pd.get_dummies(data_encoded, columns=['categorie_professionnelle'], drop_first=True)  # One-hot encoding

    # Calculer et afficher la matrice de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice de corrélation des variables')
    plt.show()

# Utilisation de la fonction
file_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\donnees_recensement_nettoyees.csv'
analyser_visualiser_donnees(file_path)
