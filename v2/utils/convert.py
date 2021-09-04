import csv
import argparse
import unicodedata
import re

import pandas as pd


def normalize_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag


def convert(input_file, output_file):
    with open(input_file) as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    converted_data = []
    for i, row in enumerate(data):
        tag_str = row["Tag"] if isinstance(row["Tag"], str) else ""
        tags = tag_str.split(';') if tag_str != "" else []
        value_str = row["Value"] if isinstance(row["Value"], str) else ""
        values = value_str.split(';') if value_str != "" else []

        if len(tags) != len(values):
            if len(tags) != 1 and len(values) != 1:
                raise ValueError("# of tags and # of values should match, or one of them should be 1. Got len(tags) = {} and len(values) = {} at row {}".format(len(tags), len(values), i))
            if len(tags) == 1:
                tags = [tags[0]] * len(values)
            else:
                values = [values[0]] * len(tags)
        
        tags = [normalize_tag(tag) for tag in tags]

        prediction = " ".join(["{tag}:{value}".format(tag=tag.replace(" ", ""), value=value.replace(" ", "")) for tag, value in zip(tags, values)])
        if prediction == "":
            prediction = "NONE"
        converted_data.append({"ID": row["ID"], "Prediction": prediction})

    with open(output_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, ["ID", "Prediction"])
        writer.writeheader()
        writer.writerows(converted_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    convert(args.input_file, args.output_file)
