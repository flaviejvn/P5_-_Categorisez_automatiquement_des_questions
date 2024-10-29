# Içi je centralise toutes les fonctions qui sont utiles au preprocessing.

import pandas as pd
from langdetect import detect
from preprocessing_config import language_names, seuil_representation_top_langage, seuil_min_rare_word, min_occurrences, top_N
from bs4 import BeautifulSoup
import contractions
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import spacy
import logging
from collections import Counter

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Charger les stopwords en anglais
stop_words = set(stopwords.words('english'))

# Charger le modèle Spacy
nlp = spacy.load('en_core_web_md')

# Fonction de nettoyage des caractères non valides.
def clean_text(text):
    if isinstance(text, str):
        # Remplacer les caractères invalides par un espace ou les ignorer
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text


def detect_language_and_analyze(df, column_name):
    """
    Fonction pour détecter la langue des textes dans une colonne d'un DataFrame et
    analyser la répartition des langues.
    
    Paramètres :
    df : pandas DataFrame
        Le DataFrame contenant les données à analyser.
    column_name : str
        Le nom de la colonne contenant les textes pour lesquels on souhaite détecter la langue.

    Retourne :
    message_detect_top_language : str
        Un message décrivant la langue la plus utilisée et sa représentation dans le DataFrame.
    language_counts : pandas Series
        La répartition des différentes langues détectées.
    most_common_language : str
        Le code de la langue la plus fréquente.
    rate : float
        Le pourcentage de textes écrits dans la langue la plus fréquente.
    """
    
    # Fonction pour détecter la langue
    def detect_language(text):
        try:
            return detect(text)
        except:
            return 'unknown'
    
    # Appliquer la détection de langue à la colonne spécifiée
    df['language'] = df[column_name].apply(detect_language)
    
    # Calculer le nombre de lignes par langue
    language_counts = df['language'].value_counts()
    
    # Détection de la langue la plus utilisée
    most_common_language = language_counts.idxmax()
    most_common_language_count = language_counts.max()
    total_count = df.shape[0]
    
    # Calcule du taux d'utilisation de la langue
    rate = (most_common_language_count / total_count) * 100
    
    # Obtenir le nom complet de la langue
    language_name = language_names.get(most_common_language, 'inconnue')
    
    # Résultat
    message_detect_top_language = (
        f"La langue la plus utilisée est '{most_common_language}' ({language_name}) "
        f"avec un taux d'utilisation de {rate:.2f}%.\n"
        f"Nombre de lignes en '{most_common_language}': {most_common_language_count} sur {total_count} lignes totales."
    )
    
    # Log du résultat
    logging.info(message_detect_top_language)
    
    # Retourner le message et les statistiques
    return message_detect_top_language, language_counts, most_common_language, rate


def filter_top_language(df, seuil_representation_top_langage, rate, most_common_language):
    """
    Filtrer un DataFrame pour ne conserver que les lignes dans la langue la plus courante
    si le taux d'utilisation de cette langue dépasse un seuil défini.

    Paramètres :
    df : pandas DataFrame
        Le DataFrame contenant les données à filtrer.
    seuil_representation_top_langage : float
        Le seuil de taux d'utilisation à partir duquel on filtre les lignes (ex: 0.75 pour 75%).
    rate : float
        Le taux d'utilisation de la langue la plus courante.
    most_common_language : str
        Le code de la langue la plus courante.

    Retourne :
    filtered_df : pandas DataFrame
        Le DataFrame filtré contenant uniquement les lignes dans la langue la plus courante si le taux le permet.
    """
    
    # Log du seuil
    logging.info(f"Seuil défini pour le filtrage : {seuil_representation_top_langage * 100:.2f}%")

    # Vérifier si le taux est suffisant pour le filtrage
    if rate > seuil_representation_top_langage:
        # Filtrer les données pour conserver uniquement les lignes dans la langue la plus courante
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

    # Log du message
    logging.info(message)
    
    # Retourner le DataFrame filtré
    return filtered_df


def preprocess_tags(df, column_name, top_N, min_occurrences):
    """
    Prétraiter une colonne de texte pour nettoyer, supprimer les doublons, filtrer les éléments rares
    et conserver uniquement les top N éléments les plus fréquents.

    Paramètres :
    df : pandas DataFrame
        Le DataFrame contenant les données à traiter.
    column_name : str
        Le nom de la colonne à traiter.
    top_N : int
        Le nombre d'éléments les plus fréquents à conserver.
    min_occurrences : int
        Le nombre minimum d'occurrences pour conserver un élément.

    Retourne :
    df : pandas DataFrame
        Le DataFrame avec une nouvelle colonne '{column_name}_cleaned' contenant les éléments prétraités.
    """
    
    # Fonction pour nettoyer les balises HTML
    def clean_text(text):
        text = text.replace("<", "").replace(">", " ")  # Enlever les balises
        text_list = text.strip().split()  # Convertir en liste d'éléments
        return text_list

    # Fonction pour supprimer les doublons
    def remove_duplicates(items_list):
        return list(set(items_list))

    # Fonction pour supprimer les éléments rares
    def remove_rare_items(items_list, rare_items):
        return [item for item in items_list if item not in rare_items]

    # Appliquer le nettoyage et suppression des doublons
    df[column_name + '_cleaned'] = df[column_name].apply(clean_text).apply(remove_duplicates)

    # Compter les occurrences des éléments
    item_counts = Counter([item for items in df[column_name + '_cleaned'] for item in items])

    # Déterminer les éléments rares (moins de min_occurrences)
    rare_items = {item for item, count in item_counts.items() if count < min_occurrences}

    # Supprimer les éléments rares
    df[column_name + '_cleaned'] = df[column_name + '_cleaned'].apply(lambda items: remove_rare_items(items, rare_items))

    # Conserver les top_N éléments les plus fréquents
    most_common_items = {item for item, count in item_counts.most_common(top_N)}

    def filter_top_items(items_list):
        return [item for item in items_list if item in most_common_items]

    # Appliquer le filtre pour conserver les top_N éléments
    df[column_name + '_cleaned'] = df[column_name + '_cleaned'].apply(filter_top_items)

    # Log du succès du prétraitement
    logging.info(f"Prétraitement de la colonne '{column_name}' terminé avec succès.")
    
    return df

# Fonction de conversion des listes en texte.
def convert_lists_to_text(df, columns):
    """
    Convertit les listes de mots en texte.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à convertir.
    columns : list
        Liste des colonnes à convertir.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes converties.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    return df


def convert_text_to_lists(df, columns):
    """
    Convertit du texte en listes de mots en utilisant un espace comme séparateur.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à convertir.
    columns : list
        Liste des colonnes à convertir.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes converties en listes de mots.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: x.split() if isinstance(x, str) else x)
    return df


# Etape de POS Tagging.
#def pos_tagging_fct(tokens):
#    """
#    Fonction pour effectuer le POS (Part-Of-Speech) tagging sur une liste de tokens en utilisant le modèle spaCy.

#    Paramètres :
#    tokens : list
#        Une liste de tokens (mots ou expressions) à étiqueter grammaticalement.

#    Retourne :
#    pos_tags : list of tuples
#        Une liste de tuples, chaque tuple contenant un mot et son étiquette grammaticale correspondante.
#        Exemple de sortie : [('This', 'DET'), ('is', 'VERB'), ('a', 'DET'), ('test', 'NOUN')]

#    Exemple d'utilisation :
#    >>> tokens = ['This', 'is', 'a', 'test']
#    >>> pos_tagging_fct(tokens)
#    [('This', 'DET'), ('is', 'VERB'), ('a', 'DET'), ('test', 'NOUN')]
#    """
#    # Création d'un document à partir de la liste de tokens.
#    doc = nlp(' '.join(tokens))
    
#    # Retourne une liste de tuples (mot, étiquette POS).
#    pos_tags = [(token.text, token.pos_) for token in doc]
#    return pos_tags
    
# Supression des balises HTML.
def clean_html(text):
    """
    Nettoie le texte en supprimant les balises HTML.

    Cette fonction utilise la bibliothèque BeautifulSoup pour analyser et extraire le texte brut 
    en supprimant les balises HTML présentes dans le texte.

    Paramètres :
    text (str) : Le texte contenant potentiellement des balises HTML à nettoyer.

    Retourne :
    str : Le texte nettoyé, avec les balises HTML supprimées.

    Exemple :
    >>> clean_html("<p>Hello, world!</p>")
    'Hello, world!'
    """
    return BeautifulSoup(text, "html.parser").get_text()


# Fonction pour compter les mots totaux et distincts dans une colonne de texte.
def count_words(text_column):
    """
    Compte le nombre total de mots et de mots distincts dans une colonne de texte.

    Paramètres:
    -----------
    text_column : pd.Series
        Colonne de texte à analyser.

    Retourne:
    --------
    tuple
        Nombre total de mots et nombre de mots distincts.
    """
    # Combiner tous les textes dans une seule chaîne
    combined_text = ' '.join(text_column)
    
    # Tokenisation du texte combiné
    tokens = word_tokenize(combined_text)
    
    # Nombre total de mots
    total_words = len(tokens)
    
    # Nombre de mots distincts
    distinct_words = len(set(tokens))
    
    return total_words, distinct_words


# Etape de POS tagging.
def pos_tagging_fct(text):
    """
    Fonction pour effectuer le POS tagging (Part-of-Speech Tagging) sur un texte donné.

    Cette fonction utilise spaCy pour analyser le texte et retourner une liste de tuples
    contenant chaque mot et son étiquette grammaticale (POS).

    Paramètres:
    -----------
    text : str
        Le texte sur lequel effectuer le POS tagging.

    Retourne:
    --------
    list
        Une liste de tuples, où chaque tuple contient un mot et son étiquette POS.
    """
    # Analyse du texte avec spaCy
    doc = nlp(text)
    
    # Extraction des étiquettes POS sous forme de liste de tuples (mot, étiquette POS)
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    return pos_tags


# Fonction pour calculer la fréquence des mots.
def get_word_frequencies(text_column):
    """
    Compte la fréquence de chaque mot dans une colonne de texte.

    Paramètres:
    -----------
    text_column : pd.Series
        Colonne de texte à analyser.

    Retourne:
    --------
    dict
        Dictionnaire des mots et leur fréquence.
    """
    # Combiner tous les textes dans une seule chaîne
    combined_text = ' '.join(text_column)
    
    # Tokenisation du texte combiné
    tokens = word_tokenize(combined_text)
    
    # Calcul de la fréquence des mots
    word_freq = Counter(tokens)
    
    return word_freq


# Fonction de préparation du texte pour le bag of words (CountVectorizer et Tf-idf) avec lemmatisation.
def transform_bow_lem_fct(text):
    """
    Transforme le texte pour l'analyse Bag of Words (CountVectorizer et Tf-idf) avec lemmatisation,
    suppression des mots rares et trop fréquents.
    
    Paramètres :
    -----------
    desc_text : str
        Le texte à transformer.
    
    Retourne :
    --------
    str
        Le texte transformé après lemmatisation, filtrage des stopwords.
    """
    
    # Tokenisation et nettoyage
    words = word_tokenize(text.lower())  # Minuscule + tokenisation
    words = [word for word in words if word.isalpha()]  # Suppression des nombres et des caractères spéciaux

    # Suppression des stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatisation
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Conversion de la liste en texte
    transf_desc_text = ' '.join(lemmatized_words)
    
    return transf_desc_text


# Fonction de préparation du texte pour le bag of words sans lemmatisation (Word2Vec)
def transform_bow_fct(desc_text):
    """
    Transforme le texte pour l'analyse Bag of Words (Word2Vec) sans lemmatisation,
    suppression des mots rares et trop fréquents.
    
    Paramètres :
    -----------
    desc_text : str
        Le texte à transformer. 
        
    Retourne :
    --------
    str
        Le texte transformé après lemmatisation, filtrage des stopwords.
    """
    
    # Tokenisation et nettoyage
    words = word_tokenize(desc_text.lower())  # Minuscule + tokenisation
    words = [word for word in words if word.isalpha()]  # Suppression des nombres et des caractères spéciaux

    # Suppression des stopwords
    words = [word for word in words if word not in stop_words]

    # Conversion de la liste en texte
    transf_desc_text = ' '.join(words)
    
    return transf_desc_text


# Fonction de préparation du texte pour le Deep Learning (USE et BERT)
def transform_dl_fct(desc_text):
    """
    Transforme le texte pour l'analyse Deep Learning (USE et BERT).

    Paramètres:
    -----------
    desc_text : str
        Le texte à transformer.

    Retourne:
    --------
    str
        Le texte transformé.
    """
    # Correction des contractions
    cleaned_text = contractions.fix(desc_text)

    # Tokenisation et nettoyage
    words = word_tokenize(cleaned_text.lower())  # Minuscule + tokenisation
    words = [word for word in words if word.isalpha()]  # Suppression des nombres et des caractères spéciaux

    # Conversion des listes en texte
    transf_desc_text = ' '.join(words)
    return transf_desc_text


# Fonction pour supprimer les mots uniques.
def remove_unique_words(words, word_frequencies):
    """
    Supprime les mots uniques d'une liste de mots.
    """
    return [word for word in words if word_frequencies[word] > 1]