import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class vader_:
    def __init__(self, text: str):
        """
        A class for performing sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool.

        This class provides methods to get the sentiment and scores of a given text.

        Args:
            text (str): The text for which sentiment analysis needs to be performed.
        """
        nltk.download('vader_lexicon')
        self.text = text
        self.scores = SentimentIntensityAnalyzer().polarity_scores(self.text)

    def get_sentiment(self) -> str:
        """
        Returns the sentiment of the text.

        Returns:
            str: The sentiment of the input text, which can be 'Positive', 'Negative', or 'Neutral'.
        """
        score = self.scores['compound']

        if score < -0.05:
            return "Negative"
        elif score > 0.05:
            return "Positive"
        else:
            return "Neutral"
        
    def get_scores(self) -> dict:
        """
        Returns the individual sentiment scores of the text.

        Returns:
            dict: A dictionary containing the scores for 'Positive', 'Negative', and 'Neutral'.
        """
        scores = {}
        scores["Positive"] = self.scores['pos']
        scores["Negative"] = self.scores['neg']
        scores["Neutral"] = self.scores['neu']

        return scores
