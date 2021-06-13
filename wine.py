from sklearn.datasets import *
from sklearn import tree

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

n_classes = 3
wine = load_wine()
clf = tree.DecisionTreeClassifier()

train_x, test_x, train_y, test_y = train_test_split(wine.data, wine.target,
                                                    test_size=0.2, random_state=666)
# binarize class labels to plot ROC
train_y = label_binarize(train_y, classes=[0, 1, 2])
test_y = label_binarize(test_y, classes=[0, 1, 2])

y_score = clf.fit(train_x, train_y).predict(test_x)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC curve for a specific class here for all classes
print(roc_auc)
