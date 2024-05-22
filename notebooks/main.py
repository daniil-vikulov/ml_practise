import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def prepare(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


@click.group()
def cli():
    pass


@click.command()
@click.option('--data', required=True, type=str, help='Path to training data CSV file')
@click.option('--test', type=str, help='Path to test data CSV file')
@click.option('--split', type=float, help='Fraction of data to be used as test set')
@click.option('--model', required=True, type=str, help='Output path for the trained model')
def train(data, test, split, model):
    df = pd.read_csv(data)
    df['text'] = df['text'].apply(prepare)
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df['text'])
    y = df['rating']

    if test:
        x_train = x
        y_train = y
        test_df = pd.read_csv(test)
        test_df['text'] = test_df['text'].apply(prepare)
        x_test = vectorizer.transform(test_df['text'])
        y_test = test_df['rating']
    elif split:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
    else:
        x_train, x_test, y_train, y_test = x, None, y, None

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)

    if x_test is not None:
        y_pred = log_reg.predict(x_test)
        print(classification_report(y_test, y_pred))

    with open(model, 'wb') as fl:
        pickle.dump((vectorizer, log_reg), fl)


@click.command()
@click.option('--model', required=True, type=str, help='Path to the trained model file')
@click.option('--data', required=True, type=str, help='Data for prediction or path to CSV file')
def predict(model, data):
    with open(model, 'rb') as f:
        vectorizer, mdl = pickle.load(f)
    try:
        df = pd.read_csv(data)
        df['text'] = df['text'].apply(prepare)
        x = vectorizer.transform(df['text'])
        predictions = mdl.predict(x)
        for pred in predictions:
            print(pred)
    except FileNotFoundError:
        prepared_text = prepare(data)
        x = vectorizer.transform([prepared_text])
        print(mdl.predict(x)[0])


cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
