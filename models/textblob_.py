from textblob import TextBlob

class textblob_:
    def __init__(self, text: str):
        """
        Initializes the textblob_ class with the provided text.

        Args:
            text (str): The text for sentiment analysis.
        """
        self.text = text
        self.blob = TextBlob(text)
    
    def get_polarity(self) -> float:
        """
        Calculates the polarity of the text.

        Returns:
            float: The polarity score of the text.
        """
        return self.blob.sentiment.polarity

    def get_sentiment(self) -> str:
        """
        Determines the sentiment of the text based on its polarity.

        Returns:
            str: The sentiment label ('Negative', 'Neutral', or 'Positive').
        """
        score = self.get_polarity()

        if score < 0:
            return "Negative"
        elif score == 0:
            return "Neutral"
        else:
            return "Positive"
    
    def get_subjectivity(self) -> float:
        """
        Calculates the subjectivity of the text.

        Returns:
            float: The subjectivity score of the text ranging [0:1] -> 0: fact, 1: public opinion
        """
        return self.blob.sentiment.subjectivity
    
    def get_scores(self) -> dict:
        return {"Sentiment": self.get_polarity(),
                "Subjectivity": self.get_subjectivity()}
