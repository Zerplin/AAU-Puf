import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = 'challenge_response_10k.csv'


def random_forest_attack():
    # load the dataset
    df = pd.read_csv(data)
    print(df.shape)

    for col in df.columns.values:
        df[col] = df[col].astype('int64')
    print(df.head())
    print(df.describe())
    print(df['64'].value_counts())

    # # Define an output image with 6 lines and 5 columns, with inside image size of 20x25:
    # fig, axs = plt.subplots(8, 8, figsize=(20, 25))  # Build the 30 histograms:
    # for i in range(8):
    #     for j in range(8):
    #         sns.histplot(data=df, x=str(i + j), hue='64', kde=True, color='skyblue', ax=axs[i, j])
    #
    # # Print the final result:
    # plt.show()
    #
    # fig, axs = plt.subplots(8, 8, figsize=(20, 25))
    # for i in range(8):
    #     for j in range(8):
    #         sns.boxplot(x=df['64'], y=df[str(i + j)], ax=axs[i, j])
    #
    # # Print the final result:
    # plt.show()

    # Define the features and output:
    y = np.array(df['64'])
    print(y)
    X = df.drop('64', axis=1)
    X = np.cumprod(np.fliplr(X), axis=1, dtype=np.int8)
    print(X)
    # Split data into train an test, with test size of 20%:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape)

    # Build the model:
    rf = RandomForestClassifier(max_depth=100, n_estimators=400, random_state=0, min_impurity_decrease=0.0000025)
    rf.fit(X_train, y_train)
    # Evaluate the model:
    y_pred = rf.predict(X_test)
    print("accuracy on training set: %.4f" % rf.score(X_train, y_train))
    print("accuracy on test set: %.4f" % rf.score(X_test, y_test))


if __name__ == '__main__':
    random_forest_attack()
