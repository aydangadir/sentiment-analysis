from models.textblob_ import textblob_
from models.vader_ import vader_
from utils import *
from preprocessing.utils import cleaning_text
from data_.utils import *
from models.feature_extraction import feature_extraction
from models.logistic_reg import log_reg

if __name__ == "__main__":
    # text = "I hate this"

    # config = import_json()

    # if config['preprocessing']['execute'] == 1:
    #     text = cleaning_text(text)

    # if config['sentiment']['model'].lower() == "textblob":
    #     sentiment_method = textblob_
    # elif config['sentiment']['model'].lower() == "vader":
    #     sentiment_method = vader_

    # sentiment = sentiment_method(text)
    # print(sentiment.get_sentiment())

    df = read_data()

    for i, row in df.iterrows():
        df.at[i, "text"] = cleaning_text(row['text'])

    feature_extractor = feature_extraction()
    df['features'] = list(feature_extractor.train(df).toarray())

    log_reg_ = log_reg()
    print(log_reg_.train(df))