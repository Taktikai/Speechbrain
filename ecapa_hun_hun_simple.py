import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import rand

with open(r"C:\Users\La Bouff Alexander\Desktop\pythonProject\ecapa\ecapaname.pickle", 'rb') as handle:
    y = pickle.load(handle)

with open(r"C:\Users\La Bouff Alexander\Desktop\pythonProject\ecapa\ecapa.pickle", 'rb') as handle2:
    x = pickle.load(handle2)

x = np.array([o[0][0].numpy() for o in x])
y = np.array(y)

# skf = StratifiedKFold(n_splits=10, shullfe=true, random_state=0)
# skf.get_n_splits(x,y)
# print(skf)

ssp = StratifiedShuffleSplit(n_splits=2, random_state=0, test_size=0.2)

for train_index, test_index in ssp.split(x, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

clf = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel="linear",C=11))
clf.fit(x_train, y_train)
# Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(gamma='auto'))])

y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
rec1 = recall_score(y_test,y_pred, pos_label=0)
rec2 = recall_score(y_test,y_pred, pos_label=1)
prec1 = precision_score(y_test,y_pred, pos_label=0)
prec2 = precision_score(y_test,y_pred, pos_label=1)
f1 = f1_score(y_test,y_pred)
print(f"confusion matrix:\n{cm}")
print(f"acc:{acc:0.3}, rec1:{rec1:0.3}, rec2:{rec2:0.3}, prec1:{prec1:0.3}, prec2:{prec2:0.3}, f1:{f1:0.3}")
