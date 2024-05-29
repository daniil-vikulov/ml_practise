import sys

import joblib


def get_data(file):
    try:
        with open(file, 'r') as file:
            return [file.read()]
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return str(e)


def detect(code_lst):
    with open('model.pkl', 'rb') as f:
        vectorizer, model = joblib.load(f)

    x_vectorized = vectorizer.transform(code_lst)

    return model.predict(x_vectorized)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    file_name = sys.argv[1]
    content = get_data(file_name)
    res = detect(content)
    print(res[0])
