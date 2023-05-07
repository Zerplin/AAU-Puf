import numpy as np  # linear algebra
import pandas as pd
from keras import Input, Model, optimizers
from keras.layers import BatchNormalization, Dense, Dropout, concatenate
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

data = 'challenge_response_10k.csv'


# data = 'Dataset_Generator/challenge_response.csv'


# baseline model
def create_baseline():
    input_layer = Input(shape=(64,))

    out1 = Dense(64, activation='relu')(input_layer)
    out1 = Dropout(0.5)(out1)
    out1 = BatchNormalization()(out1)

    out2 = Dense(64, activation='relu')(input_layer)
    out2 = Dropout(0.5)(out2)
    out2 = BatchNormalization()(out2)

    out3 = Dense(64, activation='relu')(input_layer)
    out3 = Dropout(0.5)(out3)
    out3 = BatchNormalization()(out3)

    merge = concatenate([out1, out2, out3])

    output = Dense(1, activation='sigmoid')(merge)

    # Compile model
    model = Model(inputs=input_layer, outputs=output)

    # summarize layers
    print(model.summary())

    # plot graph
    # plot_model(model, to_file='MODEL.png')

    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
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
