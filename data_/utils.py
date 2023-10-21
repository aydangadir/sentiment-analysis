from utils import import_json
import pandas

def read_data():
    """
    Reads data from a specified path and creates a pandas DataFrame.

    This function reads data from a file path specified in the configuration
    and creates a pandas DataFrame with two columns, 'text' and 'sentiment'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the read data.
    """
    df = pandas.DataFrame(columns=['text', 'sentiment'])

    config = import_json()

    path = config['data']['labeled_data_path']
    file = open(path, 'r')
    line = file.readline()

    while line:
        line = line.strip()

        sentiment = int(line[-1])
        text = line[:-1].strip()
        df.loc[len(df)] = [text, sentiment]

        line = file.readline()

    return df