import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib


with open('features_train1', 'rb') as fp:
    features_train = pickle.load(fp)
with open('labels_train1', 'rb') as fp:
    labels_train = pickle.load(fp)

X_train = np.array(features_train)
y_train = np.array(labels_train)


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train,y_train)

joblib.dump(clf,'trained_1.pkl')

#print(res.tolist().count(1))
#print(labels_test.count(1))
res = clf.score(X_train,y_train)
print(res)
print('Training Successful!')
