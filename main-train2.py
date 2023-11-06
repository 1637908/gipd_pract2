import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


MODEL_PATH = env_var = os.environ["MODEL_PATH"]

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# mymodel = np.poly1d(np.polyfit(train_x, train_y, 7)) 
mymodel = LinearRegression()
mymodel.fit(train_x.reshape(-1,1), train_y)
print("Model trained successfully")

# r2 = r2_score(test_y, mymodel(test_x))
# r2 = r2_score(test_y, mymodel.predict(test_x.reshape(-1,1))

# print("Model Score:", r2)

np.save(MODEL_PATH, mymodel)
