from models.textblob_ import textblob_
from models.vader_ import vader_
from utils import *
from preprocessing.utils import cleaning_text

if __name__ == "__main__":
    text = "I hate this"

    config = import_json()

    if config['preprocessing']['execute'] == 1:
        text = cleaning_text(text)

    if config['sentiment']['model'].lower() == "textblob":
        sentiment_method = textblob_
    elif config['sentiment']['model'].lower() == "vader":
        sentiment_method = vader_

    sentiment = sentiment_method(text)
    print(sentiment.get_sentiment())