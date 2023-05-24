import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def attack(folder_path):
    logging.basicConfig(filename='log/pufEx.log', encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    for filename in sorted(os.listdir(folder_path)):
        logging.info(filename)
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            y = df.iloc[:, -1]
            X = df
            del X[X.columns[-1]]
            X = np.cumprod(np.fliplr(X), axis=1, dtype=np.int8)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.debug(f'{X_train.shape} {y_train.shape}')

            # MLP
            from sklearn.neural_network import MLPClassifier
            clfMLP = MLPClassifier(random_state=0, learning_rate='adaptive', early_stopping=True)
            mlpt1 = time.perf_counter()
            clfMLP.fit(X_train, y_train)
            mlpt2 = time.perf_counter()
            y_pred = clfMLP.predict(X_test)
            mlpt3 = time.perf_counter()
            logging.info(
                f'Multilayer Perceptron (MLP) Fitting time: {mlpt2 - mlpt1:.4f}, Prediction time: {mlpt3 - mlpt2:.4f}, Train set score: {clfMLP.score(X_train, y_train):.4f}, Test set score: {clfMLP.score(X_test, y_test):.4f}')

            # RF
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(max_depth=100, n_estimators=250, random_state=0,
                                         min_impurity_decrease=0.0000025)
            rft1 = time.perf_counter()
            clf.fit(X_train, y_train)
            rft2 = time.perf_counter()
            y_pred = clf.predict(X_test)
            rft3 = time.perf_counter()
            logging.info(
                f'Random Forest (RF) Fitting time: {rft2 - rft1:.4f}, Prediction time: {rft3 - rft2:.4f}, Train set score: {clf.score(X_train, y_train):.4f}, Test set score: {clf.score(X_test, y_test):.4f}')

            # decision tree
            from sklearn.tree import DecisionTreeClassifier
            DTclf = DecisionTreeClassifier(max_depth=100, random_state=0, min_impurity_decrease=0.000035)
            DTt1 = time.perf_counter()
            DTclf.fit(X_train, y_train)
            DTt2 = time.perf_counter()
            y_pred = DTclf.predict(X_test)
            DTt3 = time.perf_counter()
            logging.info(
                f'Decision Trees (DT) Fitting time: {DTt2 - DTt1:.4f}, Prediction time: {DTt3 - DTt2:.4f}, Train set score: {DTclf.score(X_train, y_train):.4f}, Test set score: {DTclf.score(X_test, y_test):.4f}')

            # SVM
            C = 1.0  # SVM regularization parameter
            from sklearn.svm import SVC
            svc = SVC(kernel='linear', gamma='auto', C=C)
            svc_t1 = time.perf_counter()
            svc.fit(X_train, y_train)
            svc_t2 = time.perf_counter()
            y_pred = svc.predict(X_test)
            svc_t3 = time.perf_counter()
            logging.info(
                f'Support Vector Machine (SVM) Training time: {svc_t2 - svc_t1:.4f}, Prediction time: {svc_t3 - svc_t2:.4f}, Train set score: {svc.score(X_train, y_train):.4f}, Test set score: {svc.score(X_test, y_test):.4f}')

            # RBF SVM
            rbf_svc = rbf_svc = SVC(kernel='rbf', gamma='auto', C=C)
            rbf_svc_t1 = time.perf_counter()
            rbf_svc.fit(X_train, y_train)
            rbf_svc_t2 = time.perf_counter()
            y_pred = rbf_svc.predict(X_test)
            rbf_svc_t3 = time.perf_counter()
            logging.info(
                f'Radial Basis Function (RBF) Support Vector Machine (SVM) Training time: {rbf_svc_t2 - rbf_svc_t1:.4f}, Prediction time: {rbf_svc_t3 - rbf_svc_t2:.4f}, Train set score: {rbf_svc.score(X_train, y_train):.4f}, Test set score: {rbf_svc.score(X_test, y_test):.4f}')

            poly_svc = SVC(kernel='poly', gamma='auto', degree=3, C=C)
            poly_svc_t1 = time.perf_counter()
            poly_svc.fit(X_train, y_train)
            poly_svc_t2 = time.perf_counter()
            y_pred = poly_svc.predict(X_test)
            poly_svc_t3 = time.perf_counter()
            logging.info(
                f'Poly Support Vector Machine (SVM) Training time: {poly_svc_t2 - poly_svc_t1:.4f}, Prediction time: {poly_svc_t3 - poly_svc_t2:.4f}, Train set score: {poly_svc.score(X_train, y_train):.4f}, Test set score: {poly_svc.score(X_test, y_test):.4f}')

            from sklearn.svm import LinearSVC
            lin_svc = LinearSVC(C=C)
            lin_svc_t1 = time.perf_counter()
            lin_svc.fit(X_train, y_train)
            lin_svc_t2 = time.perf_counter()
            y_pred = lin_svc.predict(X_test)
            lin_svc_t3 = time.perf_counter()
            logging.info(
                f'Linear Support Vector Machine (SVM) Training time: {lin_svc_t2 - lin_svc_t1:.4f}, Prediction time: {lin_svc_t3 - lin_svc_t2:.4f}, Train set score: {lin_svc.score(X_train, y_train):.4f}, Test set score: {lin_svc.score(X_test, y_test):.4f}')

            # Logistic regression
            from sklearn.linear_model import LogisticRegression
            LRclf = LogisticRegression(random_state=0)
            LRt1 = time.perf_counter()
            LRclf.fit(X_train, y_train)
            LRt2 = time.perf_counter()
            y_pred = LRclf.predict(X_test)
            LRt3 = time.perf_counter()
            logging.info(
                f'Logistic Regression (LR) Training time: {LRt2 - LRt1:.4f}, Prediction time: {LRt3 - LRt2:.4f}, Train set score: {LRclf.score(X_train, y_train):.4f}, Test set score: {LRclf.score(X_test, y_test):.4f}')


if __name__ == '__main__':
    attack('PUFdataset')
