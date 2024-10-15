import pandas as pd
import numpy as np
from langdetect import detect
from data_preprocessing_config import language_names, seuil_representation_top_langage, seuil_min_rare_word, top_4_pos_tags
import contractions
import re
import unicodedata
import string
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load('en_core_web_md')
from collections import Counter

# Fonction pour détecter la langue utilisée.
def detect_language_and_analyze(df, column_name):
    """
    Détecte la langue du texte dans une colonne donnée d'un DataFrame et analyse
    la répartition des langues. Retourne un message décrivant la langue la plus
    utilisée, ainsi que les statistiques correspondantes.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant la colonne à analyser.
    column_name : str
        Le nom de la colonne contenant le texte dont la langue doit être détectée.

    Retourne:
    --------
    message_detect_top_language : str
        Un message indiquant la langue la plus utilisée et son taux de représentation.
    language_counts : pd.Series
        Un objet Series avec les langues détectées comme index et leurs occurrences comme valeurs.
    most_common_language : str
        La langue la plus courante dans la colonne spécifiée.
    rate : float
        Le pourcentage d'occurrences de la langue la plus courante par rapport au nombre total de lignes.
    """
    def detect_language(text):
        try:
            return detect(text)
        except:
            return 'unknown'
    
    df['language'] = df[column_name].apply(detect_language)
    language_counts = df['language'].value_counts()
    most_common_language = language_counts.idxmax()
    most_common_language_count = language_counts.max()
    total_count = df.shape[0]
    rate = (most_common_language_count / total_count) * 100
    language_name = language_names.get(most_common_language, 'inconnue')
    
    message_detect_top_language = (
        f"La langue la plus utilisée est '{most_common_language}' ({language_name}) "
        f"avec un taux d'utilisation de {rate:.2f}%.\n"
        f"Nombre de lignes en '{most_common_language}': {most_common_language_count} sur {total_count} lignes totales."
    )
    
    return message_detect_top_language, language_counts, most_common_language, rate


# Fonction pour filtrer les lignes selon la langue la plus courante.
def filter_top_language(df, seuil_representation_top_langage, rate, most_common_language):
    """
    Filtre les lignes d'un DataFrame pour ne conserver que celles où la langue la plus
    courante est utilisée, si cette langue dépasse un seuil de représentation.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame à filtrer.
    seuil_representation_top_langage : float
        Le seuil de représentation minimal (en pourcentage) pour que la langue la plus courante soit utilisée
        comme critère de filtrage.
    rate : float
        Le pourcentage d'occurrences de la langue la plus courante par rapport au nombre total de lignes.
    most_common_language : str
        La langue la plus courante dans la colonne spécifiée.

    Retourne:
    --------
    filtered_df : pd.DataFrame
        Le DataFrame filtré, contenant uniquement les lignes dans la langue la plus courante si le seuil est atteint.
    """
    print(f"Seuil défini pour le filtrage : {seuil_representation_top_langage * 100:.2f}%")

    if rate > seuil_representation_top_langage:
        filtered_df = df[df['language'] == most_common_language]
        message = "Taux top langage suffisant pour créer un filtre afin de ne conserver que les top langages."
    else:
        filtered_df = df
        message = "Taux top langage trop faible pour créer un filtre afin de ne conserver que les top langages."

    print(message)
    return filtered_df


# Fonction de nettoyage des tags
def clean_tags(tag_string):
    """
    Nettoie, tokenize, et convertit en minuscules une chaîne de tags sans
    décomposer les mots à l'intérieur des chevrons.
    Supprime les doublons et les tags contenant uniquement des chiffres ou des caractères spéciaux.

    Paramètres:
    -----------
    tag_string : str
        La chaîne de caractères représentant les tags.

    Retourne:
    --------
    list
        Liste de tokens nettoyés, sans doublons, sans chiffres ni caractères spéciaux.
    """
    if isinstance(tag_string, str):
        # Extraction des tags entre "<>" en tant qu'unités distinctes.
        tag_string = re.findall(r'<(.*?)>', tag_string)

        # Filtre les tags qui contiennent uniquement des chiffres ou des caractères spéciaux.
        filtered_tokens = [token for token in tag_string if re.search(r'[a-zA-Z]', token)]
        
        # Conversion en minuscules et suppression des espaces inutiles.
        cleaned_tokens = [token.lower().strip() for token in filtered_tokens]
        
        # Suppression des doublons en convertissant en set puis retour à la liste.
        unique_tokens = list(set(cleaned_tokens))
        
        return filtered_tokens
    
    return []


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


# Fonction de préparation du texte pour le bag of words (CountVectorizer et Tf-idf) avec lemmatization
# Fonction de préparation du texte pour le bag of words (CountVectorizer et Tf-idf) avec lemmatisation.
def transform_bow_lem_fct(desc_text, rare_words):
    """
    Transforme le texte pour l'analyse Bag of Words (CountVectorizer et Tf-idf) avec lemmatisation
    et supprime les mots rares spécifiés.

    Paramètres:
    -----------
    desc_text : str
        Le texte à transformer.
    rare_words : list
        Liste des mots à supprimer.

    Retourne:
    --------
    str
        Le texte transformé après lemmatisation, filtrage des stopwords et suppression des mots.
    """
    # Suppression des caractères spéciaux.
    cleaned_text = remove_special_characters(pd.DataFrame({'text': [desc_text]}), ['text'])['text'].iloc[0]

    # Tokenisation, conversion en minuscules, suppression des espaces avant/après mots.
    word_tokens = word_tokenize(cleaned_text.lower().strip())

    # Suppression des stopwords.
    filtered_words = [word for word in word_tokens if word not in stop_words]

    # Lemmatisation.
    lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in filtered_words]

    # Suppression des mots rares.
    filtered_words_no_rare = [word for word in lemmatized_words if word not in rare_words]

    # Conversion de la liste en texte.
    transf_desc_text = ' '.join(filtered_words_no_rare)
    
    return transf_desc_text


# Fonction de préparation du texte pour le bag of words sans lemmatization (Word2Vec)
def transform_bow_fct(desc_text):
    """
    Transforme le texte pour l'analyse Bag of Words (Word2Vec) sans lemmatisation.

    Paramètres:
    -----------
    desc_text : str
        Le texte à transformer.

    Retourne:
    --------
    str
        Le texte transformé après lemmatisation et autres traitements.
    """
    # Correction des contractions
    cleaned_text = contractions.fix(desc_text)

    # Suppression des caractères spéciaux
    cleaned_text = remove_special_characters(pd.DataFrame({'text': [cleaned_text]}), ['text'])['text'].iloc[0]

    # Tokenisation, conversion en minuscules suppression des espaces avant/après mots.
    word_tokens = word_tokenize(cleaned_text.lower().strip())

    # POS tagging
    pos_tags = pos_tagging_fct(' '.join(word_tokens))

    # Reconnaissance d'entités nommées (NER)
    named_entities = ner_fct(' '.join(word_tokens))

    # Suppression des stopwords
    filtered_words = [word for word in word_tokens if word not in stop_words]

    # Lemmatisation
    # lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in filtered_words]

    # Conversion des listes en texte
    transf_desc_text = ' '.join(filtered_words)
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

    # Suppression des caractères spéciaux
    cleaned_text = remove_special_characters(pd.DataFrame({'text': [cleaned_text]}), ['text'])['text'].iloc[0]

    # Tokenisation, conversion en minuscules suppression des espaces avant/après mots.
    word_tokens = word_tokenize(cleaned_text.lower().strip())

    # Conversion des listes en texte
    transf_desc_text = ' '.join(word_tokens)
    return transf_desc_text


# Fonction qui permet de récupérer des informations sur les tokens.
def display_token_info(tokens):
    """display info about corpus """
    
    # Info sur le nombre de tokens (total) et nombre de tokens uniques.
    print(f"nb tokens {len(tokens)}, nb tokens uniques {len(set(tokens))}")
    
    # Affiche les 30 premiers tokens avec leur fréquence.
    print(list(tokens[:30]))


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


# Fonction pour supprimer les balises HTML au début ou à la fin de la chaîne de caractères.
def remove_html_tags_edges(text_string):
    """
    Supprime les balises HTML seulement si elles sont présentes au début ou à la fin de la chaîne de caractères.

    Paramètres:
    -----------
    text_string : str
        Le texte à nettoyer.

    Retourne:
    --------
    str
        Le texte nettoyé avec les balises HTML en début ou en fin de chaîne supprimées.
    """
    if isinstance(text_string, str):
        # Suppression des balises HTML au début de la chaîne
        text_string = re.sub(r'^<[^>]+>', '', text_string) # cherche et supprime les balises HTML qui apparaissent au début de la chaîne (indiqué par ^).
        
        # Suppression des balises HTML à la fin de la chaîne
        text_string = re.sub(r'<[^>]+>$', '', text_string) # cherche et supprime les balises HTML qui apparaissent à la fin de la chaîne (indiqué par $).
        
        return text_string
    
    return text_string


# Fonction pour supprimer les mots dont le POS tagging ne correspond pas aux étiquettes dans la liste top_5_pos_tags.
def filter_by_pos_tags(text):
    """
    Filtre les mots en ne conservant que ceux ayant un POS tagging dans 'top_5_pos_tags'.

    Paramètres:
    -----------
    text : str
        Le texte sur lequel effectuer le POS tagging et filtrer les mots.

    Retourne:
    --------
    str
        Le texte filtré ne contenant que les mots avec les POS tags souhaités.
    """
    # POS tagging du texte
    pos_tags = pos_tagging_fct(text)
    
    # Filtrer les mots dont les POS sont dans la liste top_5_pos_tags
    filtered_words = [word for word, pos in pos_tags if pos in top_4_pos_tags]
    
    # Reconstruire la chaîne de caractères avec les mots filtrés
    return ' '.join(filtered_words)
################################################################################ ok ##############################################################

# Fonction de nettoyage du texte (correction des contractions).
def clean_text(df, columns):
    """
    Nettoie le texte en corrigeant les contractions.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à nettoyer.
    columns : list
        Liste des colonnes à nettoyer.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes nettoyées.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: contractions.fix(str(x)) if isinstance(x, str) else x)
    return df


# Fonction de suppression des caractères spéciaux.
def remove_special_characters(df, columns):
    """
    Supprime les caractères spéciaux, les URL, les balises HTML, la ponctuation, les séquences numériques
    et effectue une normalisation Unicode du texte.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à traiter.
    columns : list
        Liste des colonnes à traiter.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes traitées.
    """
    for col in columns:
        if col in df.columns:
            # Suppresion des séquences numériques.
            df[col] = df[col].apply(lambda x: re.sub(r'\b\d+\b', '', str(x)) if isinstance(x, str) else x)
            # Suppresion des séquences entre crochets.
            df[col] = df[col].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)) if isinstance(x, str) else x)
            # Suppresion des URL.
            df[col] = df[col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', str(x)) if isinstance(x, str) else x)
            # Suppresion des balises HTML.
            df[col] = df[col].apply(lambda x: re.sub('<.*?>', '', str(x)) if isinstance(x, str) else x)
            # Suppresion de la ponctuation.
            df[col] = df[col].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', str(x)) if isinstance(x, str) else x)
            # Suppresion des nouvelles lignes.
            df[col] = df[col].apply(lambda x: re.sub('\n', '', str(x)) if isinstance(x, str) else x)
            # Normalisation des caractères Unicode.
            df[col] = df[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)) if isinstance(x, str) else x)
            # Remplace les tirets par des espaces
            df[col] = df[col].apply(lambda x: re.sub('-', ' ', str(x)) if isinstance(x, str) else x)
        else:
            print(f"Avertissement: La colonne '{col}' n'existe pas dans le DataFrame.")
    return df




# Fonction de tokenisation et conversion en minuscules.
def tokenize_and_lowercase(df, columns):
    """
    Tokenise le texte et le convertit en minuscules.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à traiter.
    columns : list
        Liste des colonnes à traiter.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes traitées.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: word_tokenize(x.lower().strip()) if isinstance(x, str) else x)
    return df


# Fonction de suppression des stopwords.
def remove_stopwords(df, columns):
    """
    Supprime les stopwords du texte.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à traiter.
    columns : list
        Liste des colonnes à traiter.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes traitées.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) else x)
    return df


# Fonction de lemmatisation.
def lemmatize_columns(df, columns):
    """
    Lemmatisation des colonnes spécifiées dans un DataFrame.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à lemmatiser.
    columns : list
        Liste des noms de colonnes à lemmatiser.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes spécifiées lemmatisées.
    """
    lemmatizer = WordNetLemmatizer()

    for col in columns:
        df[col] = df[col].apply(lambda words: [lemmatizer.lemmatize(w) for w in words] if isinstance(words, list) else words)
    
    return df


# Fonction de suppression des mots rares.
def remove_rare_words(df, columns, seuil_min_rare_word=1, output_file="mots_supprimes.csv"):
    """
    Supprime les mots rares des colonnes spécifiées.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à traiter.
    columns : list
        Liste des colonnes à traiter.
    seuil_min_rare_word : int, facultatif
        Seuil minimum pour qu'un mot soit considéré comme non-rare.
    output_file : str, facultatif
        Chemin du fichier .csv pour enregistrer les mots supprimés.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les mots rares supprimés.
    """
    all_words = []
    for col in columns:
        all_words.extend([word for words in df[col] for word in words if isinstance(words, list)])
    
    word_counter = Counter(all_words)
    rare_words = [word for word, count in word_counter.items() if count <= seuil_min_rare_word]

    # Enregistrer les mots rares dans un fichier CSV
    pd.DataFrame(rare_words, columns=["Word"]).to_csv(output_file, index=False)
    
    for col in columns:
        df[col] = df[col].apply(lambda words: [word for word in words if word not in rare_words] if isinstance(words, list) else words)
    
    return df


# Fonction de lemmatisation utilisant spaCy.
def lemmatize_with_spacy(df, columns):
    """
    Effectue la lemmatisation des textes en utilisant spaCy.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les colonnes à lemmatiser.
    columns : list
        Liste des colonnes à traiter.

    Retourne:
    --------
    pd.DataFrame
        Le DataFrame avec les colonnes traitées.
    """
    for col in columns:
        df[col] = df[col].apply(lambda text: " ".join([token.lemma_ for token in nlp(text)]) if isinstance(text, str) else text)
    
    return df


# Etape de reconnaissance d'entités nommées (NER).
def ner_fct(text):
    """
    Fonction pour effectuer la Reconnaissance d'Entités Nommées (Named Entity Recognition, NER) sur un texte donné.

    Cette fonction utilise spaCy pour identifier et classer les entités nommées dans un texte.
    Elle retourne une liste de tuples contenant chaque entité nommée et son type.

    Paramètres:
    -----------
    text : str
        Le texte sur lequel effectuer la reconnaissance d'entités nommées.

    Retourne:
    --------
    list
        Une liste de tuples, où chaque tuple contient une entité nommée et son label NER.
    """
    # Analyse du texte avec spaCy
    doc = nlp(text)
    
    # Extraction des entités nommées sous forme de liste de tuples (entité, label NER)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return named_entities
