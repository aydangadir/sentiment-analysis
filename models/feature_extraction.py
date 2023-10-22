from utils import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import pandas as pd

class feature_extraction:
    def __init__(self):
        """
        A class for feature extraction from text data.

        This class provides methods to extract features from text data
        using various techniques such as CountVectorizer and TfidfVectorizer.
        """
        self.feature_extractor = None
        self.config = import_json()

        self.path = self.config['sentiment']['path'] + self.config['sentiment']['feature_extractor'] + ".pkl"
        self.feature_extractor = self.load_model()

    def count_vectorizer(self, df: pd.DataFrame):
        """
        Performs feature extraction using the CountVectorizer technique.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            The feature-extracted data.
        """
        self.feature_extractor = CountVectorizer()
        x_count = self.feature_extractor.fit_transform(df['text'])
        return x_count

    def tfidf_vectorizer(self, df):
        """
        Performs feature extraction using the TfidfVectorizer technique.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            The feature-extracted data.
        """
        self.feature_extractor = TfidfVectorizer()
        x_count = self.feature_extractor.fit_transform(df['text'])
        return x_count

    def train(self, df):
        """
        Trains the feature extraction model based on the chosen technique.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            The trained feature extraction model.
        """
        if self.config['sentiment']['feature_extractor'].lower() == "count_vectorizer":
            return self.count_vectorizer(df)
        elif self.config['sentiment']['feature_extractor'].lower() == "tfidf_vectorizer":
            return self.tfidf_vectorizer(df)

    def transform(self, text):
        """
        Transforms the input text using the feature extraction model.

        Args:
            text: The text data to be transformed.

        Raises:
            Exception: If the Feature Extractor file is not found.

        Returns:
            The transformed text data.
        """
        if self.feature_extractor is None:
            raise Exception("Feature Extractor file not found :/")

        return self.feature_extractor.transform(text)

    def save_model(self):
        """Saves the feature extraction model to a pickle file."""
        with open(self.path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)

    def load_model(self):
        """Loads the feature extraction model from a pickle file."""
        if not if_exists(self.path):
            return None

        with open(self.path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
