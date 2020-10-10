# import libraries
import os
import argparse
import pandas as pd
import numpy as np
import re
import pathlib
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import joblib
import nltk


def get_project_path():
    """
    this function get project absolute path regardless of we the python script being executed.
    relative path for loading data or model can be define give project absolute path
    return  project absolute path
    :return:
    """
    if len(__file__.split("/")) > 1:
        project_path = str(pathlib.Path(__file__).parent.parent.absolute())
    else:
        project_path = ".."
    return project_path


def load_data(database_filepath):
    """
    load stored preprocessed data from sqlite database given path to be use for generating plots and analysis.
    returns train data, train labels, list of class/category names
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
    receives a text message and breaks it down to relevant tokens using NLP techniques
    the resulting word array will be used for feature extraction in classification pipeline
    return array of tokens
    :param text:
    :return:
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = nltk.tokenize.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if tok.lower() not in nltk.corpus.stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok, pos="v").lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model_simple():
    """
    This function helps in building the model.
    Creating the pipeline

    Return the model
    :return:
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    return pipeline


def build_model():
    """
    This function helps in building the model.
    Creating the pipeline
    Applying Grid search
    Return the model
    """
    # creating pipeline
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(AdaBoostClassifier())),
        ]
    )

    # parameters
    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__max_df": (0.8, 1.0),
        "vect__max_features": (None, 10000),
        "clf__estimator__n_estimators": [50, 100],
        "clf__estimator__learning_rate": [0.1, 1.0],
    }

    # grid search
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)

    return cv


def build_optimized_model():
    """
     This function helps in building the model.
    Creating the pipeline
    Applying Grid search
    Return the model
    :return:
    """
    clf = MultiOutputClassifier(SVC())
    tune_parameters = {
        "clf_estimator__gamma": [1e-1, 1e-2, 1e-3],
        "clf_estimator__C": [1, 10, 100],
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__max_df": (0.8, 1.0),
        "vect__max_features": (None, 10000),
    }
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", clf),
        ]
    )
    clf_grid = GridSearchCV(
        estimator=pipeline, n_jobs=-1, cv=3, param_grid=tune_parameters
    )
    return clf_grid


def display_evaluation_results(y_test, y_pred, label_names):
    """
    applies classification metrics on predicted classes and prints out f1 score, accuracy and confusion matrix per class
    :param y_test:
    :param y_pred:
    :param label_names
    :return:
    """
    for i, l_name in enumerate(label_names):
        labels = np.unique(y_pred)
        target_names = ["".join(["not ", l_name]), l_name]
        print(
            classification_report(
                y_pred=y_pred[:, i],
                y_true=y_test.iloc[:, i].to_numpy(),
                labels=labels,
                target_names=target_names,
            )
        )
        print("")
    print(
        "average accuracy {}".format(
            sum(
                [
                    accuracy_score(y_test.iloc[:, i].to_numpy(), y_pred[:, i])
                    for i in range(y_pred.shape[1])
                ]
            )
            / y_pred.shape[1]
        )
    )
    print(
        "average f1_score {}".format(
            sum(
                [
                    f1_score(y_test.iloc[:, i].to_numpy(), y_pred[:, i])
                    for i in range(y_pred.shape[1])
                ]
            )
            / y_pred.shape[1]
        )
    )


def evaluate_model(model, X_test, Y_test, category_names):
    """
    runs evaluation on test data and displays the results
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    y_pred = model.predict(X_test)
    display_evaluation_results(Y_test, y_pred, category_names)


def save_model(model, model_filepath, model_name="dr_trained_model.lzma"):
    """
    saves trained model in given path
    :param model:
    :param model_filepath:
    :param model_name:
    :return:
    """
    # save
    m_f = "".join([model_filepath, model_name])
    if os.path.exists(m_f):
        os.remove(m_f)
    joblib.dump(value=model, filename=m_f, compress=("lzma", 9))


def generate_arg_parser():
    """
    this function receives input arguments for various functions.

    :return:
    """
    project_path = get_project_path()
    # load data
    default_db_path = "".join([project_path, "/data/DisasterResponseDataBase.db"])
    default_model_path = "".join([str(project_path), "/models/"])

    parser = argparse.ArgumentParser(
        description="Load data from database and train classifier and dump the trained model."
    )

    parser.add_argument(
        "--db_file",
        action="store",
        dest="db_file",
        type=str,
        default=default_db_path,
        help="Path to disaster response database",
    )

    parser.add_argument(
        "--model_file",
        action="store",
        dest="model_file",
        type=str,
        default=default_model_path,
        help="path to store trained machine leaning model.",
    )
    return parser.parse_args(), parser


def main():
    args_params, parser = generate_arg_parser()
    if not args_params.db_file or not args_params.model_file:
        parser.print_help()
        exit(1)
    print("\n Downloading required NLTK libraries....\n")
    nltk.download(["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"])
    print("Loading data...\n    DATABASE: {}".format(args_params.db_file))
    X, Y, category_names = load_data(args_params.db_file)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # build/train/evaluate/save model
    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    print("Saving model...\n    MODEL: {}".format(args_params.model_file))
    save_model(
        model,
        args_params.model_file,
    )

    # build/train/evaluate/save optimized model
    print("Building optimized model...")
    opt_model = build_optimized_model()

    print("Training optimized model...")
    opt_model.fit(X_train, Y_train)

    print("Evaluating optimized model...")
    evaluate_model(opt_model, X_test, Y_test, category_names)

    print("Saving optimized model...\n    MODEL: {}".format(args_params.model_file))
    save_model(opt_model, args_params.model_file, "dr_trained_opt_model.lzma")

    print("Trained model saved!")


if __name__ == "__main__":
    main()
