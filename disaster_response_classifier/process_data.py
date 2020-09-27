# import libraries
import pandas as pd
from sqlalchemy import create_engine
import os
import argparse


def load_data(messages_filepath, categories_filepath):
    """

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

    :param df:
    :return:
    """
    # Split categories into separate category columns.
    df = pd.concat([df, df.categories.str.split(";", expand=True)], axis=1).drop(
        columns=["categories"]
    )

    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = df.iloc[0, 4 : df.shape[1]].apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    df.rename(
        columns=dict(zip(df.columns[4 : df.shape[1]], pd.Index(category_colnames))),
        inplace=True,
    )

    # set each value to be the last character of the string
    # convert column from string to numeric
    df[df.columns[4 : df.shape[1]]] = (
        df[df.columns[4 : df.shape[1]]].applymap(lambda x: x[-1]).astype(int)
    )

    # drop duplicates
    # drop nan
    df.dropna(subset=["message"], inplace=True)
    df.drop_duplicates(subset="message", inplace=True)
    return df


def save_data(df, database_filepath):
    """

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
    parser = argparse.ArgumentParser(
        description="Process row data and store in database."
    )

    parser.add_argument(
        "--msg_file",
        action="store",
        dest="msg_file",
        type=argparse.FileType("r"),
        help="path to disaster response messages file.",
    )

    parser.add_argument(
        "--cat_file",
        action="store",
        dest="cat_file",
        type=argparse.FileType("r"),
        help="path to disaster response messages categories file.",
    )

    parser.add_argument(
        "--db_file",
        action="store",
        dest="db_file",
        type=str,
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
