import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import train_test_split

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import models
from tensorflow.python.keras.models import load_model

data_df = pd.read_csv("survey lung cancer.csv")
GENDER = "GENDER"
AGE = "AGE"
SMOKING = "SMOKING"
CHRONIC_DISEASE = "CHRONIC DISEASE"
YELLOW_FINGERS = "YELLOW_FINGERS"
ANXIETY = "ANXIETY"
PEER_PRESSURE = "PEER_PRESSURE"
FATIGUE = "FATIGUE"
ALLERGY = "ALLERGY"
WHEEZING = "WHEEZING"
ALCOHOL_CONSUMING = "ALCOHOL_CONSUMING"
COUGHING = "COUGHING"
SHORTNESS_OF_BREATH = "SHORTNESS_OF_BREATH"
SWALLOWING_DIFFICULTY = "SWALLOWING_DIFFICULTY"
CHEST_PAIN = "CHEST_PAIN"
LUNG_CANCER = "LUNG_CANCER"

for col in data_df.columns:
    if col != AGE:
        data_df[col] = data_df[col].astype('category').cat.codes
data_df.head(5)

X = data_df.iloc[:, 0:15]
y = data_df.iloc[:, -1]
y = y.astype(int)
X = X.astype(int)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

model = Sequential()
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# train model: epoch, batch_size, validation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=100, validation_data=(X_val, y_val))

model.save("mymodel.h5")