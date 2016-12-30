# question: can mammographic mass data be used to predict whether patients have benign or malignant tumours?

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix

def plotcatfeats(dataframe, feature):
    feat_table = pd.crosstab(index=dataframe['Severity'], columns=dataframe[feature])
    feat_table.plot(kind='bar', figsize=(12,8), title=feature, stacked=False)

if __name__ == "__main__":
    # load data
    # using the Mammographic Mass Data from UCI repository 
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
    names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
    data = pd.read_csv(url, names=names)

    # print dataset characteristics
    print data.shape # 961 samples, 6 columns
    print data.head() # first five rows
    print data.dtypes # gives the data type for each column
    description = data.describe() 
    print description
    
    # convert Age data type to numeric
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    
    #print number of samples with benign (0) or malignant (1) tumours
    print data['Severity'].value_counts() 
    
    # plot histograms of age against tumour type
    data.hist(column='Age', by='Severity', figsize=(12,8))
    
    # plot boxplots of age against tumous type
    data.boxplot(column='Age', by='Severity', figsize=(12,8))
    
    # plot catagorical features by tumour severity
    plotcatfeats(data, 'Shape')
    plotcatfeats(data, 'Margin')
    plotcatfeats(data, 'Density')

    # split data into train and test sets
    train_set = data.sample(frac=0.8)
    test_set = data.drop(train_set.index)