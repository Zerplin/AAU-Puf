import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = 'challenge_response_10k.csv'


def mlp_attack():
    df = pd.read_csv(data)
    print(df.shape)

    for col in df.columns.values:
        df[col] = df[col].astype('int64')

    print(df.head())
    print(df.describe())
    print(df['64'].value_counts())

    y = df['64']
    X = df.drop('64', axis=1)
    X = np.cumprod(np.fliplr(X), axis=1, dtype=np.int8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    clf_mlp = MLPClassifier(random_state=0, early_stopping=False)
    clf_mlp.fit(X_train, y_train)

    y_pred = clf_mlp.predict(X_test)

    print('Model accuracy score with criterion gini index: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    # print the scores on training and test set

    print('Training set score: {:.4f}'.format(clf_mlp.score(X_train, y_train)))

    print('Test set score: {:.4f}'.format(clf_mlp.score(X_test, y_test)))


if __name__ == '__main__':
    mlp_attack()