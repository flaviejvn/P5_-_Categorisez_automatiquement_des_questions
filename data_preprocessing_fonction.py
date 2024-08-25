from langdetect import detect
from data_preprocessing_config import language_names, seuil_representation_top_langage

# Fonction pour détecter la langue utilisée.
def detect_language_and_analyze(df, column_name):
    # Fonction pour détecter la langue.
    def detect_language(text):
        try:
            return detect(text)
        except:
            return 'unknown'
    
    # Appliquer la détection de langue à la colonne spécifiée
    df['language'] = df[column_name].apply(detect_language)
    
    # Calculer le nombre de lignes par langue
    language_counts = df['language'].value_counts()
    
    # Détection de la langue la plus utilisée.
    most_common_language = language_counts.idxmax()
    most_common_language_count = language_counts.max()
    total_count = df.shape[0]
    
    # Calcule du taux d'utilisation de la langue.
    rate = (most_common_language_count / total_count) * 100
    
    # Obtention du nom complet de la langue.
    language_name = language_names.get(most_common_language, 'inconnue')
    
    # Résultat.
    message_detect_top_language = (
        f"La langue la plus utilisée est '{most_common_language}' ({language_name}) "
        f"avec un taux d'utilisation de {rate:.2f}%.\n"
        f"Nombre de lignes en '{most_common_language}': {most_common_language_count} sur {total_count} lignes totales."
    )
    
    # Retourne le message et les statistiques associées.
    return message_detect_top_language, language_counts, most_common_language, rate


def filter_top_language(df, seuil_representation_top_langage, rate, most_common_language):
    # Afficher le seuil en début de traitement
    print(f"Seuil défini pour le filtrage : {seuil_representation_top_langage * 100:.2f}%")

    # Vérifier si le taux est suffisant pour le filtrage
    if rate > seuil_representation_top_langage:
        # Filtrer les données pour ne conserver que celles correspondant à la langue la plus courante
        filtered_df = df[df['language'] == most_common_language]
        message = (
            "Taux top langage suffisant pour créer un filtre afin de ne conserver que les top langages."
        )
    else:
        # Si le taux est insuffisant, ne pas filtrer
        filtered_df = df
        message = (
            "Taux top langage trop faible pour créer un filtre afin de ne conserver que les top langages."
        )

    # Afficher le message de confirmation
    print(message)
    
    # Retourner le DataFrame filtré
    return filtered_df

#def filter_top_language(df, threshold=seuil_representation_top_langage):
#    if rate > threshold:
#        filtered_df = df[df['language'] == most_common_language]
#        return filtered_df
#    else:
#        return df