
# load the packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

data = load_iris()  # load the data
df = pd.DataFrame(data.data, columns=data.feature_names)  # convert to a dataframe
df['Species'] = data.target  # create a species column
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# extract & name the features and target variables
x = df.drop(columns="Species")
y = df["Species"]
feature_names = x.columns
labels = y.unique()

# split the dataset
X_train, test_x, y_train, test_lab = train_test_split(x, y, test_size=0.4, random_state=42)

# fit the model to the training data
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# plot the tree figure
plt.figure(figsize=(15, 10), facecolor='w')
a = tree.plot_tree(clf,
                   feature_names=feature_names,
                   class_names=labels,
                   rounded=True,
                   filled=True,
                   fontsize=14)
plt.show()

# export the decision rules which show how the model behaved.
# The first split is based on petal length. <2.45cm is iris-setosa,
# >2.45cm is iris-virginica. A further split occurs for those with petal length >2.45cm
# with two further splits to end up with more accurate final classifications
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)

# we're not as interested in the training set as we are in the performance on the test set.
# We are interested in how this performs in terms of
#   - true positives (predicted true and actually true),
#   - false positives (predicted true but not actually true),
#   - false negatives (predicted false but actually true) and
#   - true negatives (predicted false and actually false).
# we will use a confusion matrix allows us to visualise how the predicted and true
# labels match up by showing predicted values on one axis and actual values on the
# other. This is useful to identify where we may get false positives or false
# negatives and hence how the algorithm has performed.
test_pred_decision_tree = clf.predict(test_x)
confusion_matrix = metrics.confusion_matrix(test_lab, test_pred_decision_tree)
matrix_df = pd.DataFrame(confusion_matrix)
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_xticklabels(['']+labels)
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(labels), rotation=0)
plt.show()

metrics.accuracy_score(test_lab, test_pred_decision_tree)
precision = metrics.precision_score(test_lab, test_pred_decision_tree, average=None)

precision_results = pd.DataFrame(precision, index=labels)
precision_results.rename(columns={0: 'precision'}, inplace=True)
print(precision_results)  # Tells us how may of the values predicted to be in a certain class are actually in that class
#                           True positive (number in diagonal)/All positives (column sum)

recall = metrics.recall_score(test_lab, test_pred_decision_tree, average=None)
recall_results = pd.DataFrame(recall, index=labels)
recall_results.rename(columns={0: 'Recall'}, inplace=True)
print(recall_results)  # tells us how many of the values in each class were given the correct label thus telling us how
# it performed relative to false negatives. True positive (number in diagonal)/All assignments (row sum)

f1 = metrics.f1_score(test_lab, test_pred_decision_tree, average=None)
f1_results = pd.DataFrame(f1, index=labels)
f1_results.rename(columns={0: 'f1'}, inplace=True)
print(f1_results)  # this is a weighted average of precision and recall, 1 being best, 0 being worst
# 2 * (precision * recall)/(precision + recall)

# we can get all of these metrics in a single output with the following piece of code:
print(metrics.classification_report(test_lab, test_pred_decision_tree))

# calculating the importance of each of the features in the final tree output. This is the total amount that the gini
# index or entropy index (gini in this case) decreases due to splits over a given feature.
importance = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 3)})
importance.sort_values('importance', ascending=False, inplace=True)
print(importance)
# shows here that Petal Length had the greatest importance as the first division was based on this. However, since
# only one decision tree has been run this does not mean that the other features are not important, only that they were
# not needed in this decision tree.

# we can also see how it responds to changes in hyperparameters by using GridSearchCV. This performs cross validation
# on the model by performing the algorithm on multiple runs of the sets of the training set, and tells us how the model
# responds.
tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5], 'min_samples_split': [2, 4, 6, 8, 10]}]
scores = ['recall']

for score in scores:
    print()
    print(f'Tuning hyperparameters for {score}')
    print()

    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring=f'{score}_macro')
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(f"{mean:0.3f} (+/-{std * 2:0.03f}) for {params}")

# For our purpose, we can change the max_depth and min_samples_split parameters which control how deep the tree goes,
# and the number of samples required to split an internal node, which tells us the best hyperparameters for this are
# max_depth = 2 and min_samples_split = 2.
