# question: can mammographic mass data be used to predict whether patients have benign or malignant tumours?

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

def plotcatfeats(dataframe, feature):
    feat_table = pd.crosstab(index=dataframe['Severity'], columns=dataframe[feature])
    feat_table.plot(kind='bar', figsize=(12,8), title=feature, stacked=False)
    
def imputenans(data, strategy):
    impute = Imputer(strategy=strategy)
    interim = data.reshape(-1, 1)
    return impute.fit_transform(interim)

if __name__ == '__main__':
    # LOAD DATA
    # using the Mammographic Mass Data from UCI repository 
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'
    names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
    data = pd.read_csv(url, names=names)

    # VISUALISE DATA
    # print dataset characteristics
    print 'Dataset size:'
    print data.shape # 961 samples, 6 columns
    print
    print 'First five rows:'
    print data.head() # first five rows
    print
    print 'Data types:'
    print data.dtypes # gives the data type for each column
    print
    print 'Data characteristics:'
    description = data.describe() 
    print description
    print
    
    # convert Age data type to numeric (missing values converted to NaN)
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data['Shape'] = pd.to_numeric(data['Shape'], errors='coerce')
    data['Margin'] = pd.to_numeric(data['Margin'], errors='coerce')
    data['Density'] = pd.to_numeric(data['Density'], errors='coerce')
    
    #print number of samples with benign (0) or malignant (1) tumours
    print 'Samples in each class:'
    print data['Severity'].value_counts() 
    print
    
    # plot histograms of age against tumour type
    data.hist(column='Age', by='Severity', figsize=(12,8))
    
    # plot boxplots of age against tumous type
    data.boxplot(column='Age', by='Severity', figsize=(12,8))
    
    # plot catagorical features by tumour severity
    plotcatfeats(data, 'Shape')
    plotcatfeats(data, 'Margin')
    plotcatfeats(data, 'Density')

    # PREPROCESS DATA
    # print pairwise correlations between features; margin and shape have strongest correlation, all features retained since only a small number of them
    print 'Pairwise Pearson correlation coefficients:'
    print data.corr(method='pearson')
    print
    
    # check proportion of NaN values in each feature column; highest is density with 8% missing, all features retained
    print 'Proportion of NaN values:'
    print 'Age: ' + str(float(data['Age'].isnull().sum())/data['Age'].isnull().count())
    print 'Shape: ' + str(float(data['Shape'].isnull().sum())/data['Shape'].isnull().count())
    print 'Margin: ' + str(float(data['Margin'].isnull().sum())/data['Margin'].isnull().count())
    print 'Density: ' + str(float(data['Density'].isnull().sum())/data['Density'].isnull().count())
    print
    
    # split data into train and test sets
    train_set = data.sample(frac=0.8)
    test_set = data.drop(train_set.index)
    
    # replace NaN age values with mean (applying averages from training set to train and test sets)
    train_set.loc[train_set['Severity']==0, 'Age'] = imputenans(train_set.loc[train_set['Severity']==0, 'Age'], 'mean')
    train_set.loc[train_set['Severity']==1, 'Age'] = imputenans(train_set.loc[train_set['Severity']==1, 'Age'], 'mean')
    
    test_set.loc[test_set['Severity']==0, 'Age'] = imputenans(test_set.loc[test_set['Severity']==0, 'Age'], 'mean')
    test_set.loc[test_set['Severity']==1, 'Age'] = imputenans(test_set.loc[test_set['Severity']==1, 'Age'], 'mean')    
    
    # replace NaN categorical feature values with mode (applying averages from training set to train and test sets)
    train_set.loc[train_set['Severity']==0, 'Shape'] = imputenans(train_set.loc[train_set['Severity']==0, 'Shape'], 'most_frequent')
    train_set.loc[train_set['Severity']==1, 'Shape'] = imputenans(train_set.loc[train_set['Severity']==1, 'Shape'], 'most_frequent')
    train_set.loc[train_set['Severity']==0, 'Margin'] = imputenans(train_set.loc[train_set['Severity']==0, 'Margin'], 'most_frequent')
    train_set.loc[train_set['Severity']==1, 'Margin'] = imputenans(train_set.loc[train_set['Severity']==1, 'Margin'], 'most_frequent')
    train_set.loc[train_set['Severity']==0, 'Density'] = imputenans(train_set.loc[train_set['Severity']==0, 'Density'], 'most_frequent')
    train_set.loc[train_set['Severity']==1, 'Density'] = imputenans(train_set.loc[train_set['Severity']==1, 'Density'], 'most_frequent')    
    
    test_set.loc[test_set['Severity']==0, 'Shape'] = imputenans(test_set.loc[test_set['Severity']==0, 'Shape'], 'most_frequent')
    test_set.loc[test_set['Severity']==1, 'Shape'] = imputenans(test_set.loc[test_set['Severity']==1, 'Shape'], 'most_frequent')
    test_set.loc[test_set['Severity']==0, 'Margin'] = imputenans(test_set.loc[test_set['Severity']==0, 'Margin'], 'most_frequent')
    test_set.loc[test_set['Severity']==1, 'Margin'] = imputenans(test_set.loc[test_set['Severity']==1, 'Margin'], 'most_frequent')
    test_set.loc[test_set['Severity']==0, 'Density'] = imputenans(test_set.loc[test_set['Severity']==0, 'Density'], 'most_frequent')
    test_set.loc[test_set['Severity']==1, 'Density'] = imputenans(test_set.loc[test_set['Severity']==1, 'Density'], 'most_frequent')
    
    # combine categories with few samples
    # combine margin categories 2 and 3
    train_set.loc[train_set['Margin']==2, 'Margin'] = 2.5
    train_set.loc[train_set['Margin']==3, 'Margin'] = 2.5
    test_set.loc[test_set['Margin']==2, 'Margin'] = 2.5
    test_set.loc[test_set['Margin']==3, 'Margin'] = 2.5
    
    # combine density categories 1 and 2
    train_set.loc[train_set['Density']==1, 'Density'] = 1.5
    train_set.loc[train_set['Density']==2, 'Density'] = 1.5
    test_set.loc[test_set['Density']==1, 'Density'] = 1.5
    test_set.loc[test_set['Density']==2, 'Density'] = 1.5
    
    # combine density categories 3 and 4
    train_set.loc[train_set['Density']==3, 'Density'] = 3.5
    train_set.loc[train_set['Density']==4, 'Density'] = 3.5
    test_set.loc[test_set['Density']==3, 'Density'] = 3.5
    test_set.loc[test_set['Density']==4, 'Density'] = 3.5
    
    # bin age into categories?

    
    # extract features and labels
    train_feats = train_set.loc[:, 'Age':'Density']
    train_labels = train_set['Severity']
    
    test_feats = test_set.loc[:, 'Age':'Density']
    test_labels = test_set['Severity']
    
    # APPLY MACHINE LEARNING ALGORITHMS
    
    # for recording f1 scores
    scores = {}
    
    # Naive Bayes
    nb = GaussianNB()
    scores_nb = cross_val_score(nb, train_feats, train_labels, cv=5, scoring='f1')
    scores['Naive Bayes'] = scores_nb.mean()

    lr = LogisticRegression()
    scores_lr = cross_val_score(lr, train_feats, train_labels, cv=5, scoring='f1')
    scores['Logistic Regression'] = scores_lr.mean()
    
    lda = LinearDiscriminantAnalysis()
    scores_lda = cross_val_score(lda, train_feats, train_labels, cv=5, scoring='f1')
    scores['Linear DA'] = scores_lda.mean()   
    
    svm = SVC()
    scores_svm = cross_val_score(svm, train_feats, train_labels, cv=5, scoring='f1')
    scores['SVM'] = scores_svm.mean()
    
    cart = tree.DecisionTreeClassifier()
    scores_cart = cross_val_score(cart, train_feats, train_labels, cv=5, scoring='f1')
    scores['CART'] = scores_cart.mean()
    
    knn = KNeighborsClassifier()
    scores_knn = cross_val_score(knn, train_feats, train_labels, cv=5, scoring='f1')
    scores['KNN'] = scores_knn.mean()
    
    rfc = RandomForestClassifier()
    scores_rfc = cross_val_score(rfc, train_feats, train_labels, cv=5, scoring='f1')
    scores['Random Forest'] = scores_rfc.mean()

    gbc = GradientBoostingClassifier()
    scores_gbc = cross_val_score(gbc, train_feats, train_labels, cv=5, scoring='f1')
    scores['Gradient Boosting'] = scores_gbc.mean()
    
    #identify best classifier
    print "Five best performing classifiers (F1-score):" 
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    for cl in top5:
        print cl + " (" + str(scores[cl]) + ")"
    
    #clf.fit(train_feats, train_labels)   
    #predictions = clf.predict(test_feats)

    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #svm = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='f1')
    #svm.fit(train_feats, train_labels)
    
    #print svm.best_params_
    #preds = svm.predict(test_feats)
    #print metrics.classification_report(test_labels, preds, target_names=None)
    
    # extract and print info about classifier performance
    #full_report = metrics.classification_report(test_labels,predictions,target_names=None)
    #print full_report