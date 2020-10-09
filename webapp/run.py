import json
import plotly
import pandas as pd
import pathlib
import argparse
import random, threading, webbrowser

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


def get_data_graph(df_counts, df_names, title, y_title, x_title):
    """

    :param df_counts:
    :param df_names:
    :param title:
    :param y_title
    :param x_title
    :return:
    """
    graphs = [
        {
            "data": [Bar(x=df_names, y=df_counts)],
            "layout": {
                "title": title,
                "yaxis": {"title": y_title},
                "xaxis": {"title": x_title},
            },
        }
    ]
    # encode plotly graphs in JSON

    return graphs


def get_graphs():
    """

    :return:
    """
    # make genre graph
    genre_counts = df.genre.value_counts()
    genre_names = list(genre_counts.index)
    genre_graph = get_data_graph(
        genre_counts,
        genre_names,
        title="Distribution of Message Genres",
        y_title="Count",
        x_title="Genre",
    )
    # make adi related graph
    aid_counts = (
        df[df.message.str.lower().str.contains("help")]
        .aid_related.value_counts()
        .reset_index()
        .aid_related
    )
    aid_names = ["Not Aid Related", "Aid Related"]
    aid_graph = get_data_graph(
        aid_counts,
        aid_names,
        title="Distribution of Aid Related Messages containing world 'help' ",
        y_title="Count",
        x_title="Aid Related Message",
    )

    medical_counts = (
        df[df.message.str.lower().str.contains("doctor")]
        .medical_help.value_counts()
        .reset_index()
        .medical_help
    )
    medical_names = ["Not Medical Help", "Medical Help"]
    medical_graph = get_data_graph(
        medical_counts,
        medical_names,
        title="Distribution of Medical Help Messages containing world 'doctor' ",
        y_title="Count",
        x_title="Medical Hel Message",
    )

    graphs = genre_graph + aid_graph + medical_graph
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    json_graph = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return ids, json_graph


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals

    # render web page with plotly graphs
    ids, json_graph = get_graphs()
    return render_template("master.html", ids=ids, graphJSON=json_graph)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    aid_links = "https://www.muenchen.de/int/en/tourism/important-information/emergency-help/most-important-emergency-numbers.html"
    fire_link = "https://www.notruf112.bayern.de/notruf112/"
    rescue_link = "https://www.seenotretter.de/en/who-we-are/"
    military_link = "https://www.bundeswehr.de/de/aktuelles/schwerpunkte/search-and-rescue-sar-fliegende-lebensretter-rettungsdienst"
    food_link = "https://www.bundesregierung.de/breg-en/news/germany-is-second-largest-donor-1547358"
    money_link = "https://www.arbeitsagentur.de/en/financial-support"
    refugee_link = "https://www.drk.de/en/aid-worldwide/what-we-do/refugee-relief/services-provided-for-refugees-by-the-grc/"
    disaster_link = "https://www.auswaertiges-amt.de/en/aussenpolitik/themen/humanitaerehilfe/cyclone-idai/2201498"
    links = {
        "request": aid_links,
        "offer": aid_links,
        "aid_related": aid_links,
        "medical_help": fire_link,
        "medical_products": fire_link,
        "search_and_rescue": rescue_link,
        "security": aid_links,
        "military": military_link,
        "water": disaster_link,
        "food": food_link,
        "shelter": rescue_link,
        "clothing": rescue_link,
        "money": money_link,
        "missing_people": rescue_link,
        "refugees": refugee_link,
        "death": aid_links,
        "other_aid": aid_links,
        "infrastructure_related": aid_links,
        "transport": fire_link,
        "buildings": fire_link,
        "electricity": fire_link,
        "tools": fire_link,
        "hospitals": aid_links,
        "shops": food_link,
        "aid_centers": aid_links,
        "other_infrastructure": disaster_link,
        "weather_related": disaster_link,
        "floods": disaster_link,
        "storm": disaster_link,
        "fire": fire_link,
        "earthquake": fire_link,
        "cold": disaster_link,
        "other_weather": disaster_link,
    }
    # This will render the go.html Please see that file.
    classification_results = zip(df.columns[4:], classification_labels, links.values())
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
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(0.75, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)


if __name__ == "__main__":
    main()
