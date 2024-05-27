import os
import click.testing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from main import cli, prepare


def test_prepare():
    text = "Here's an example: Testing, one, two, three"
    expected_output = "here exampl test one two three"
    assert prepare(text) == expected_output

    text = "Although there flight was generally ok, there were some drawbacks"
    expected_output = "although flight general ok drawback"
    assert prepare(text) == expected_output


def test_train():
    model_path = "temp_model.pkl"
    runner = click.testing.CliRunner()
    # noinspection PyTypeChecker
    result = runner.invoke(cli, ['train', '--data', "../data/train.csv", '--model', model_path, '--split', '0.2'])

    assert result.exit_code == 0
    assert os.path.exists(model_path)

    if os.path.exists(model_path):
        os.remove(model_path)


def test_predict():
    model_path = "model.pkl"

    review = "Although there flight was generally ok, there were some drawbacks"

    runner = click.testing.CliRunner()
    # noinspection PyTypeChecker
    result = runner.invoke(cli, ['predict', '--model', model_path, '--data', review])

    assert result.exit_code == 0


def test_predict_empty_data():
    model_path = "model.pkl"

    review = ""

    runner = click.testing.CliRunner()
    # noinspection PyTypeChecker
    result = runner.invoke(cli, ['predict', '--model', model_path, '--data', review])

    assert result.exit_code == 0


def test_predict_wierd_input():
    model_path = "model.pkl"

    review = "_#@__)(__()__)(__@#_"

    runner = click.testing.CliRunner()
    # noinspection PyTypeChecker
    result = runner.invoke(cli, ['predict', '--model', str(model_path), '--data', review])

    assert result.exit_code == 0


def test_split():
    df = pd.DataFrame({
        'text': ['Good', 'Bad', 'Average', 'Excellent'],
        'rating': [4, 1, 3, 5]
    })
    df['text'] = df['text'].apply(prepare)
    bow = TfidfVectorizer()
    x = bow.fit_transform(df['text'])
    y = df['rating']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    assert x_train.shape[0] == x_test.shape[0]
