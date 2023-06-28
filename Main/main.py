import csv
import os
import numpy as np
import scipy.sparse as sps
from argparse import ArgumentParser
from Functions.eals import ElementwiseAlternatingLeastSquares, load_model
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


def load_ratings():
    # Load training data and build a sparse matrix for ratings
    print("Loading the training data")
    with open(os.path.join(BASE_DIR, "..", "Seg2Rating", "my_rating.csv"), newline="", encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = []
        cols = []
        vals = []
        for line in reader:
            rows.append(int(line["userId"]))
            cols.append(int(line["objectId"]))
            vals.append(float(line["rating"]))

    ratings = sps.csr_matrix(
        (vals, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.float32
    )
    return ratings, rows, cols


def parse_args():
    # Parse command line arguments
    parser = ArgumentParser(description="Example recommender based on the MovieLens 20M dataset")
    subparsers = parser.add_subparsers(dest="subcommand")
    parser_fit = subparsers.add_parser("fit", help="Fit the model")
    parser_fit.add_argument(
        "--num_iter", type=int, default=50, help="Number of training iterations"
    )

    return parser.parse_args()


def fit(args):
    # Load ratings and fit the EALS model
    ratings, rows, cols = load_ratings()
    print("Fitting the model")
    model = ElementwiseAlternatingLeastSquares()
    model.fit(ratings, show_loss=True)
    print(f"Saving the model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done")
    return rows, cols


def main():
    args = parse_args()
    rows, cols = fit(args)
    model = load_model(MODEL_PATH)

    user_vector = model.user_factors
    pred_ratings = (model.item_factors @ user_vector.T).T

    # Normalize and clip predicted ratings
    pred = (pred_ratings + 0.5).astype(int)
    pred[pred > 5] = 5

    # Calculate difference between predicted and actual ratings
    draw = abs(pred - np.loadtxt('../Seg2Rating/gd.txt'))
    draw[np.array(rows, int), np.array(cols, int)] = -1
    print(draw.sum() / (draw.shape[0] * draw.shape[1] - len(rows)))

    # Visualize difference using a heatmap
    plt.figure(dpi=300)
    plt.imshow(draw, 'gray')
    plt.colorbar()
    plt.savefig('show.png')
    plt.show()


if __name__ == "__main__":
    main()
