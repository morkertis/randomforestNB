from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.seterr(divide='ignore', invalid='ignore')

# Naive Bayes Random Forest - Random Forest model that contains Naive Bayes Decision Trees    
class NBRandomForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=NB_DecisionTreeClassifier(),  ## need to change
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

# Naive Bayes Decision Tree - Decision Tree model that in is leaves has naive bayes classifier 
class NB_DecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False):
        super(NB_DecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)
        self.NB_clf = {}

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        super(NB_DecisionTreeClassifier, self).fit(X, y, sample_weight=sample_weight, check_input=check_input,
                                                   X_idx_sorted=X_idx_sorted)
        self.NBclf(X, y)

    def NBclf(self, X, y):
        X = np.array(X)
        y = np.array(y)
        indices = {}
        leaves = self.apply(X)
        for idx, l in enumerate(leaves):
            if l not in indices.keys():
                indices[l] = [idx]
            else:
                indices[l].append(idx)
        for key in indices:
            clf = GaussianNB()
            xi = X[indices[key], :]
            yi = np.ravel(y[indices[key]])
            clf.fit(xi, yi)
            self.NB_clf[key] = clf

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_outputs_))
        leaves = self.apply(X)
        for idx, l in enumerate(leaves):
            if l not in self.NB_clf.keys():
                predictions[idx] = super(DecisionTreeClassifier, self).predict([X[idx]])
            else:
                pred = self.NB_clf[l].predict([X[idx]])
                if pred == np.nan:
                    pred = super(DecisionTreeClassifier, self).predict([X[idx]])
                predictions[idx] = pred
        return np.ravel(predictions)

    def predict_proba(self, X, check_input=True):
        X = np.array(X)
        proba = []
        leaves = self.apply(X)
        for idx, l in enumerate(leaves):
            classes = np.zeros(len(self.classes_))
            if l not in self.NB_clf.keys():
                proba[idx] = super(DecisionTreeClassifier, self).predict_proba([X[idx]])
            else:
                pred = self.NB_clf[l].predict([X[idx]])
                itemindex = np.where(self.classes_ == pred)
                classes[itemindex] = 1
                #                if pred == np.nan:
                #                    pred=super(DecisionTreeClassifier, self).predict_proba([X[idx]])
                proba.append(classes)
        return proba


def main():
    
    # =============================================================================
    #  test the new model vs regular random forest    
    # =============================================================================

    data = pd.read_csv("data/fish.csv")
    # Split dataset into training set and test set
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values

    print(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    RFclf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0 ,min_samples_leaf=10)
    RFclf.fit(X_train, y_train)
    y_pred = RFclf.predict(X_test)
    print("Accuracy RF:", metrics.accuracy_score(y_test, y_pred))
    print("balanced_accuracy_score RF:", metrics.balanced_accuracy_score(y_test, y_pred))
    
    clfNB = NBRandomForestClassifier(criterion='gini', n_estimators=100, random_state=0, min_samples_leaf=10)
    clfNB.fit(X_train, y_train)
    y_predNB = clfNB.predict(X_test)
    print("Accuracy RFNB:", metrics.accuracy_score(y_test, y_predNB))
    print("balanced_accuracy_score RFNB:", metrics.balanced_accuracy_score(y_test, y_predNB))


if __name__ == '__main__':
    main()


