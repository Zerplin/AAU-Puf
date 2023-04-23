import numpy as np  # linear algebra
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

data = 'challenge_response_10k.csv'
# data = 'Dataset_Generator/challenge_response.csv'


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(64, input_shape=(64,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def attack():
    # load the dataset
    df = pd.read_csv(data)
    print(df.shape)

    for col in df.columns.values:
        df[col] = df[col].astype('int64')

    print(df.head())
    print(df.describe())
    print(df['64'].value_counts())

    # split into input (x) and output (y) variables
    # Define the features and output:
    y = df['64']
    X = df.drop('64', axis=1)
    X = np.cumprod(np.fliplr(X), axis=1, dtype=np.int8)

    # encode class values as integers
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # encoded_y = encoder.transform(y)
    encoded_y = y

    # evaluate model with standardized dataset
    estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, encoded_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


if __name__ == '__main__':
    attack()
