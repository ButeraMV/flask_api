from flask import Flask, jsonify
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import json

app = Flask(__name__)

data = fetch_movielens(min_rating=4.0)

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

def recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        movies_to_send = []

        for x in top_items[:10]:
            movies_to_send.append(x)

        return movies_to_send

@app.route('/')
def index():
    to_display = recommendation(model, data, [1])
    return jsonify(to_display)

if __name__ == '__main__':
    app.run(debug=True)
