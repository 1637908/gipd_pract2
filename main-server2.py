import os

from flask import Flask, request
import numpy as np
from sklearn.linear_model import LinearRegression

MODEL_PATH = env_var = os.environ["MODEL_PATH"]

app = Flask(__name__)


@app.route("/")
def model_documentation():
    return """
<h1>Welcome to customer spent prediction model</h1>

<p>Please use our api to use the model:</p>
<p>curl localhost:5000/model?minutes=5</p>
"""


@app.route("/model")
def model():
    minutes = request.args.get('minutes')
    model = LinearRegression(np.load(MODEL_PATH))
    minutes = np.array([int(minutes)]).reshape(-1,1)
    return {"spent": model.predict(minutes)}
   
