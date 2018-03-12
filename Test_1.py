import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib

with open('features_test', 'rb') as fp:
    features_test = pickle.load(fp)
with open('labels_test', 'rb') as fp:
    labels_test = pickle.load(fp)

clf = joblib.load('trained_1.pkl')

X_test = np.array(features_test)
y_test = np.array(labels_test)

res = clf.predict(X_test)

print(res.tolist().count(1))
print(labels_test.count(1))
print(clf.score(features_test,labels_test))
