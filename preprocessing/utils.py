import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from utils import import_json

def delete_punctuation(text: string):
    """
    Removes all punctuation characters from the input text.

    Args:
        text (str): The text from which punctuation needs to be removed.

    Returns:
        str: The input text without any punctuation characters.
    """

    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def delete_stop_words(text: string, additional_stopwords: list = None):
    """
    Removes stop words from the input text.

    Args:
        text (str): The text from which stop words need to be removed.
        additional_stopwords (list, optional): A list of additional stop words to be removed. Defaults to None.

    Returns:
        str: The input text without any stop words.
    """
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    if additional_stopwords is not None:
        stop_words += additional_stopwords
    
    word_tokens = word_tokenize(text)
    # converts the words in word_tokens to lower case and then checks whether 
    # they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    
    filtered_sentence = " ".join(filtered_sentence)
    return filtered_sentence

def noisy_data(text: string):
    # depends on the case
    return False

def expand_contractions(text: string):
    """
    Expands contractions in the input text. (such as "I'll -> I will", "can't -> cannot", etc)

    Args:
        text (str): The text in which contractions need to be expanded.

    Returns:
        str: The input text with expanded contractions.
    """
    return contractions.fix(text)

def stemming(text: string):
    """
    Applies stemming to the words in the input text.

    Stemming reduces words to their word stem, base, or root form.

    Args:
        text (str): The text in which words need to be stemmed.

    Returns:
        str: The input text with words converted to their stemmed forms.
    """
    word_tokens = word_tokenize(text)
    porter_stemmer = PorterStemmer()
    stems = [porter_stemmer.stem(w) for w in word_tokens]
    stems = " ".join(stems)
    return stems

def lemmatize(text: string):
    """
    Applies lemmatization to the words in the input text.

    Lemmatization reduces words to their base or dictionary form, 
    which is known as the lemma.

    Args:
        text (str): The text in which words need to be lemmatized.

    Returns:
        str: The input text with words converted to their lemmatized forms.
    """
    word_tokens = word_tokenize(text)
    wordnet_lemmatizer = WordNetLemmatizer()
    lems = [wordnet_lemmatizer.lemmatize(w) for w in word_tokens]
    lems = " ".join(lems)
    return lems

def cleaning_text(text: string, additional_stopwords: list = None):
    """
    Applies a series of text cleaning operations to the input text.

    This function performs various text cleaning operations, such as
    removing noisy data, deleting punctuation, expanding contractions,
    removing stop words, and either stemming or lemmatizing the words
    based on the configuration provided.

    Args:
        text (str): The text to be cleaned.
        additional_stopwords (list, optional): A list of additional stop words to be removed. Defaults to None.

    Returns:
        str: The cleaned text after performing all specified cleaning operations, or None if noisy data is detected.
    """
    if noisy_data(text):
        return None
    
    text = expand_contractions(text)
    text = delete_punctuation(text)
    text = delete_stop_words(text, additional_stopwords=additional_stopwords)
    
    config = import_json()

    if config['preprocessing']['stem_lem'].lower() == "stemming":
        text = stemming(text)
    elif config['preprocessing']['stem_lem'].lower() == "lemmatization":
        text = lemmatize(text)

    return text