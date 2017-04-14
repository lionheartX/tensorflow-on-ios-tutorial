import numpy as np
import pandas as pd

df = pd.read_csv("voice.csv", header=0)

# convert "male" to 1, "female" to 0
labels = (df["label"] == "male").values * 1
labels = labels.reshape(-1, 1) 
del df["label"]

data = df.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 123456)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)