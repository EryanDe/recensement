import pandas as pd
import re

def nettoyer_donnees(file_path, output_path):
    # Charger les données
    data = pd.read_csv(file_path)

    # 1. Suppression des doublons
    data = data.drop_duplicates()

    # 2. Correction des erreurs typographiques dans la colonne "sexe"
    # Assurer que les valeurs de "sexe" sont bien 'M' ou 'F'
    data['sexe'] = data['sexe'].str.strip().str.upper()
    data = data[data['sexe'].isin(['M', 'F'])]

    # 3. Correction des erreurs typographiques dans "categorie_professionnelle"
    # Suppression des espaces inutiles et correction des capitalisations
    data['categorie_professionnelle'] = data['categorie_professionnelle'].str.strip().str.title()

    # 4. Correction des caractères spéciaux dans toutes les colonnes de type texte
    def corriger_caracteres_speciaux(texte):
        # Remplacement des caractères spéciaux par des lettres sans accent, ou suppression s'ils sont invalides
        return re.sub(r'[^\w\s]', '', texte)

    # Appliquer la correction des caractères spéciaux à toutes les colonnes de type 'object' (texte)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].apply(lambda x: corriger_caracteres_speciaux(x) if isinstance(x, str) else x)

    # 5. Sauvegarde des données nettoyées dans un nouveau fichier CSV
    data.to_csv(output_path, index=False)

    return data

# Utilisation de la fonction
file_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\recensement.csv'
output_path = 'C:\\Users\\Eryan\\Desktop\\recensement\\donnees_recensement_nettoyees.csv'

donnees_nettoyees = nettoyer_donnees(file_path, output_path)

# Afficher un aperçu des données nettoyées
print(donnees_nettoyees.head())
