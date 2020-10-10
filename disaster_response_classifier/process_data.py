# import libraries
import pandas as pd
from sqlalchemy import create_engine
import os
import argparse
import pathlib


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


def load_data(messages_filepath, categories_filepath):
    """
    load message and category from given csv files
    returns concatenated  dataframe
    :param messages_filepath:
    :param categories_filepath:
    :return:
    """
    return pd.concat(
        [
            pd.read_csv(messages_filepath),
            pd.read_csv(categories_filepath)["categories"],
        ],
        axis=1,
    )


def clean_data(df):
    """
    Split categories into separate category columns.
    select the first row of the categories dataframe
    use this row to extract a list of new column names for categories.
    one way is to apply a lambda function that takes everything
    up to the second to last character of each string with slicing
    rename the columns of `categories`
    set each value to be the last character of the string
    convert column from string to numeric
    drop duplicates
    drop nan
    returns clean dataframe
    :param df:
    :return:
    """

    df = pd.concat([df, df.categories.str.split(";", expand=True)], axis=1).drop(
        columns=["categories"]
    )

    category_colnames = df.iloc[0, 4 : df.shape[1]].apply(lambda x: x[0:-2])

    df.rename(
        columns=dict(zip(df.columns[4 : df.shape[1]], pd.Index(category_colnames))),
        inplace=True,
    )

    df[df.columns[4 : df.shape[1]]] = (
        df[df.columns[4 : df.shape[1]]]
        .applymap(lambda x: x[-1] if x[-1] in ["0", "1"] else "1")
        .astype(int)
    )

    df.dropna(subset=["message"], inplace=True)
    df.drop_duplicates(subset="message", inplace=True)
    return df


def save_data(df, database_filepath):
    """
    saves given dataframe in given path as sqlite database
    :param df:
    :param database_filepath:
    :return:
    """

    # Save the clean dataset into an sqlite database.
    database_filename = "".join([database_filepath, "DisasterResponseDataBase.db"])
    if os.path.exists(database_filename):
        os.remove(database_filename)
    engine = create_engine("".join(["sqlite:///", database_filename]))
    df.to_sql("DisasterResponseData", engine, index=False)
    engine.dispose()


def generate_arg_parser():
    """
    this function receives input arguments for various functions.

    :return:
    """
    project_path = get_project_path()
    # load data
    default_db_path = "".join([project_path, "/data/"])
    default_cat_file = "".join(
        [str(project_path), "/data/disaster-response-categories.csv"]
    )
    default_msg_file = "".join(
        [str(project_path), "/data/disaster-response-messages.csv"]
    )

    parser = argparse.ArgumentParser(
        description="Process row data and store in database."
    )

    parser.add_argument(
        "--msg_file",
        action="store",
        dest="msg_file",
        default=default_msg_file,
        type=argparse.FileType("r"),
        help="path to disaster response messages file.",
    )

    parser.add_argument(
        "--cat_file",
        action="store",
        dest="cat_file",
        default=default_cat_file,
        type=argparse.FileType("r"),
        help="path to disaster response messages categories file.",
    )

    parser.add_argument(
        "--db_file",
        action="store",
        dest="db_file",
        type=str,
        default=default_db_path,
        help="path to SQLLite database file for storing processed data.",
    )
    return parser.parse_args()


def main():
    pars_args = generate_arg_parser()

    print(
        "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
            pars_args.msg_file, pars_args.cat_file
        )
    )
    df = load_data(pars_args.msg_file, pars_args.cat_file)

    print("Cleaning data...")
    df = clean_data(df)

    print("Saving data...\n    DATABASE: {}".format(pars_args.db_file))
    save_data(df, pars_args.db_file)

    print("Cleaned data saved to database!")


if __name__ == "__main__":
    main()
