import pickle
from sklearn import tree
import numpy as np
from sklearn.externals import joblib


with open('features_train1', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train1', 'rb') as fp:
    labels_train = pickle.load(fp)

print(len(features_train))
print(len(labels_train))

X_train = np.array(features_train)
y_train = np.array(labels_train)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

joblib.dump(clf,'trained_2.pkl')

res = clf.score(X_train,y_train)
print(res)
print('Training Successful!')
