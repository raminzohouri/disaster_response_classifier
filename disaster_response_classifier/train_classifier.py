# import libraries
import os
import pandas as pd
import argparse
from sqlalchemy import create_engine
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])


def load_data(database_filepath):
    """

    :param database_filepath:
    :return:
    """
    engine = create_engine("".join(["sqlite:///", database_filepath]))
    table_name = "".join([database_filepath.split("/")[-1], "Table"])
    df = pd.read_sql_query("select * from DisasterResponseData", con=engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4 : df.shape[1]]
    category_names = df.columns[4 : df.shape[1]].to_list()
    return X, Y, category_names


def tokenize(text):
    """

    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """

    :return:
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(KNeighborsClassifier())),
        ]
    )
    return pipeline


def display_evaluation_results(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :return:
    """
    labels = np.unique(y_pred)
    # confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    # print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    y_pred = model.predict(X_test)
    display_evaluation_results(Y_test, y_pred)


def save_model(model, model_filepath):
    """

    :param model:
    :param model_filepath:
    :return:
    """
    # save
    m_f = "".join([model_filepath, "dr_trained_model.lzma"])
    if os.path.exists(m_f):
        os.remove(m_f)
    joblib.dump(value=model, filename=m_f, compress=("lzma", 9))


def generate_arg_parser():
    """

    :return:
    """
    parser = argparse.ArgumentParser(
        description="Load data from database and train classifier and dump the trained model."
    )

    parser.add_argument(
        "--db_file",
        action="store",
        dest="db_file",
        type=str,
        help="Path to disaster response database",
    )

    parser.add_argument(
        "--model_file",
        action="store",
        dest="model_file",
        type=str,
        help="path to store trained machine leaning model.",
    )
    return parser.parse_args(), parser


def main():
    args_params, parser = generate_arg_parser()
    if not args_params.db_file or not args_params.model_file:
        parser.print_help()
        exit(1)

    print("Loading data...\n    DATABASE: {}".format(args_params.db_file))
    X, Y, category_names = load_data(args_params.db_file)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    print("Saving model...\n    MODEL: {}".format(args_params.model_file))
    save_model(model, args_params.model_file)

    print("Trained model saved!")


if __name__ == "__main__":
    main()
