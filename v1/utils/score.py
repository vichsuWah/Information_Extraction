import csv
import argparse
import unicodedata
import re


def normalize_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag


def score(ref_file, pred_file):
    with open(ref_file) as csvfile:
        reader = csv.DictReader(csvfile)
        ref_data = list(reader)

    with open(pred_file) as csvfile:
        reader = csv.DictReader(csvfile)
        pred_data = list(reader)

    f_score = 0.0
    for ref_row, pred_row in zip(ref_data, pred_data):
        refs = set(ref_row["Prediction"].split())
        preds = set(pred_row["Prediction"].split())

        p = len(refs.intersection(preds)) / len(preds) if len(preds) > 0 else 0.0
        r = len(refs.intersection(preds)) / len(refs) if len(refs) > 0 else 0.0
        f = 2*p*r / (p+r) if p + r > 0 else 0
        f_score += f

    return f_score / len(ref_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file")
    parser.add_argument("pred_file")

    args = parser.parse_args()

    s = score(args.ref_file, args.pred_file)
    print(s)
