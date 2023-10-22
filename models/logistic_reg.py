from utils import *
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class log_reg:
    def __init__(self):
        """
        A class for Logistic Regression model.

        This class provides methods for training and predicting using a
        Logistic Regression model.
        """
        self.log_reg = None
        self.config = import_json()

        self.path = self.config['sentiment']['path'] + "log_reg.pkl"
        self.log_reg = self.load_model()

    def train(self, df: pd.DataFrame) -> float:
        """
        Trains the Logistic Regression model.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data and corresponding labels.

        Returns:
            float: The accuracy score of the trained model.
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['features'].tolist(), df['sentiment'], test_size=0.2, random_state=42)

        # Training the logistic regression model
        self.log_reg = LogisticRegression()
        self.log_reg.fit(X_train, y_train)

        y_pred = self.log_reg.predict(X_test)
        self.save_model()

        return accuracy_score(y_test, y_pred)

    def predict(self, X):
        """
        Predicts using the Logistic Regression model.

        Args:
            X: The input data to be predicted.

        Raises:
            Exception: If the Logistic Regression model is not found.

        Returns:
            The predicted values.
        """
        if self.log_reg is None:
            raise Exception("Logistic Regression not found :/")

        return self.log_reg.predict(X)

    def save_model(self):
        """Saves the Logistic Regression model to a pickle file."""
        create_folder(self.config['sentiment']['path'])
        with open(self.path, 'wb') as f:
            pickle.dump(self.log_reg, f)

    def load_model(self):
        """Loads the Logistic Regression model from a pickle file."""
        if not if_exists(self.path):
            return None

        with open(self.path, 'rb') as f:
            self.log_reg = pickle.load(f)
