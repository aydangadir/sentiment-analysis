from utils import *
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class svm_:
    def __init__(self):
        """
        A class for Support vector machine model.
        This class provides methods for training and predicting using a
        Support vector machine model.
        """
        self.svm = None
        self.config = import_json()

        self.path = self.config['sentiment']['path'] + "svm.pkl"
        self.svm = self.load_model()

    def train(self, df: pd.DataFrame) -> float:
        """
        Trains the Support vector machine model.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data and corresponding labels.

        Returns:
            float: The accuracy score of the trained model.
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['features'].tolist(), df['sentiment'], test_size=0.2, random_state=42)

        # Training the Support vector machine model
        self.svm = svm.SVC(kernel='linear')
        self.svm.fit(X_train, y_train)

        y_pred = self.svm.predict(X_test)
        self.save_model()

        return accuracy_score(y_test, y_pred)

    def predict(self, X):
        """
        Predicts using the Support vector machine model.

        Args:
            X: The input data to be predicted.

        Raises:
            Exception: If the Support vector machine model is not found.

        Returns:
            The predicted values.
        """
        if self.svm is None:
            raise Exception("Support vector machine not found :/")

        return self.svm.predict(X)

    def save_model(self):
        """Saves the Support vector machine model to a pickle file."""
        create_folder(self.config['sentiment']['path'])
        with open(self.path, 'wb') as f:
            pickle.dump(self.svm, f)

    def load_model(self):
        """Loads the Support vector machine model from a pickle file."""
        if not if_exists(self.path):
            return None

        with open(self.path, 'rb') as f:
            self.svm = pickle.load(f)
