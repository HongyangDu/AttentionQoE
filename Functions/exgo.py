import numpy as np
import os
import csv
import scipy.sparse as sps
from functions.eals import ElementwiseAlternatingLeastSquares, load_model


def load_ratings(BASE_DIR, file):
    print("Loading the training data")
    with open(os.path.join(BASE_DIR, file), newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        cols = []
        vals = []
        for line in reader:
            # if float(line["rating"]) > 3:
            rows.append(int(line["userId"]))
            cols.append(int(line["movieId"]))
            # vals.append(1.0)
            vals.append(float(line["rating"]))

    ratings = sps.csr_matrix(
        (vals, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.float32
    )
    return ratings


if __name__ == "__main__":
    ratings = load_ratings()
    # batch training
    user_items = sps.csr_matrix(ratings, dtype=np.float32)
    model = ElementwiseAlternatingLeastSquares(factors=1)
    model.fit(user_items)

    # learned latent vectors
    res1 = model.user_factors
    res2 = model.item_factors
    s = np.dot(res1, res2.T)
    # print(model.user_items)
    np.savetxt('ger.txt', s)
    # print(s)

    # nm = 'ger.rating'
    # f = open(nm, 'w')
    # f.write(str(s))
    # f.close()

#
# # online training for new data (user_id, item_id)
# model.update_model(1, 0)
#
# # rating matrix and latent vectors will be expanded for a new user or item
# model.update_model(0, 5)
#
# # current rating matrix
# model.user_items
#
# # save and load the model
# model.save("model.joblib")
# model = load_model("model.joblib")
