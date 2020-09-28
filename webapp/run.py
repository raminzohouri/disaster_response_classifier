import json
import plotly
import pandas as pd
import pathlib
import argparse

# from ..disaster_response_classifier import utils
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)
df = None
model = None


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


def get_project_path():
    """

    :return:
    """
    if len(__file__.split("/")) > 1:
        project_path = str(pathlib.Path(__file__).parent.parent.absolute())
    else:
        project_path = ".."
    return project_path


def load_dataframe(db_file):
    """

    :param db_file:
    :return:
    """
    engine = create_engine("".join(["sqlite:///", db_file]))
    df = pd.read_sql_table("DisasterResponseData", engine)
    return df


def load_model(model_file):
    """

    :param model_file:
    :return:
    """
    with open(model_file, "rb") as pickle_file:
        model = joblib.load(pickle_file)
    return model


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.genre.value_counts()
    genre_names = list(genre_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def generate_arg_parser():
    """

    :return:
    """
    project_path = get_project_path()
    # load data
    default_db_path = "".join([project_path, "/data/DisasterResponseDataBase.db"])
    default_model_path = "".join([str(project_path), "/models/dr_trained_model.lzma"])

    parser = argparse.ArgumentParser(
        description="Load data from database, load model, and run the webapp."
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
    global df, model
    df = load_dataframe(args_params.db_file)
    model = load_model(args_params.model_file)
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
