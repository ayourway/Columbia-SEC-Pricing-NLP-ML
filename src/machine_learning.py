import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score, GridSearchCV


# Data processing
def data_processing():
    dependencies = pd.read_csv("NLP_cb_NormDoc_diff_table.csv")
    dependencies = dependencies.dropna()#axis=0, how = 'all')
    dependencies = dependencies[np.isfinite(dependencies['Negative'])]
    dependencies = dependencies[np.isfinite(dependencies['OneMonth_y'])]

    # del dependencies["Industry_Benchmark"]
    dependencies.iloc[np.where(dependencies ==np.inf)] = 1e6
    dependencies.iloc[np.where(dependencies ==-np.inf)] = -1e6
    
    dependencies = pd.DataFrame.as_matrix(self=dependencies)
    X_y = dependencies[:, 6:]
    X_y = pd.DataFrame(X_y,columns=None)
    X_y = resample(X_y, random_state=0)

    # Generate the training set
    train_X_y = X_y.sample(frac = 0.8, random_state = 1)

    # Select anything not in the training set
    test_X_y = X_y.loc[~X_y.index.isin(train_X_y.index)]

    # Prepare for train and test dataset
    X_y_matrix = pd.DataFrame.as_matrix(self=train_X_y)
    train_X_y = pd.DataFrame.as_matrix(self=train_X_y)
    test_X_y = pd.DataFrame.as_matrix(self=test_X_y)
    remove = (X_y_matrix[:, -2] > 1)


    train_X = train_X_y[:, 6:-5]
    train_y_regression = train_X_y[train_X_y[:, -2] > 1, :]
    train_y_regression = train_y_regression[:,-2]
    train_y = train_X_y[:, -1]
    train_y = label(train_y)

    test_X = test_X_y[:, 6:-5]
    test_y_regression = test_X_y[test_X_y[:, -2] > 1, :]
    test_y_regression = test_y_regression[:, -2]
    test_y = test_X_y[:, -1]
    test_y = label(test_y)

    #Dimension reduction by PCA
    pca= PCA(n_components = 10)
    train_X_low = pca.fit_transform(train_X)
    test_X_low = pca.fit_transform(test_X)

    #return train_X_low,test_X_low,train_y,test_y
    return train_X,test_X,train_y,test_y,train_y_regression,test_y_regression

def label(y):
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def accuarcy(prediction, test_y):
    right = 0
    for i in range(len(prediction)):
        if prediction[i] == test_y[i]:
            right+=1
    return right/len(prediction)

def random_forest(train_X,test_X,train_y,test_y):
    predictions_list = []
    # initialize the random forest model
    for i in range(1,50):
        model = RandomForestClassifier(n_estimators= 5, min_samples_leaf= i, random_state=1)
    
    # Initialize Random Forest Model with GridSearchCV for parameters selection
    #RFC = RandomForestClassifier()
    #parameters = {'n_estimators':[5,10,25,50,60],'min_samples_leaf':[6], 'random_state':[1]}
    #model = GridSearchCV(RFC,parameters)

    # Fit the model to the data
        model.fit(train_X, list(map(int,train_y)))
    # Make predictions.
        predictions = model.predict(test_X)
    # Compute the error.
        predictions_list.append(accuarcy(predictions, test_y))

    return max(predictions_list) , predictions_list.index(max(predictions_list)) + 1

# Change y, PCA, RandomForest parameters

def KNN(train_X,test_X,train_y,test_y):
    predictions_list = []
    for i in range(1,50):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_X, list(map(int,train_y)))
        predictions = neigh.predict(test_X)
        predictions_list.append(accuarcy(predictions, test_y))
    return max(predictions_list), predictions_list.index(max(predictions_list)) + 1

# Change y, PCA, KNN parameters
def GBclassifer(train_X,test_X,train_y,test_y):
    predictions_list = []
    for i in range(1,50):
        model = GradientBoostingClassifier(n_estimators=i,max_depth=2)
        model.fit(train_X, list(map(int,train_y)))
        predictions = model.predict(test_X)
        predictions_list.append(accuarcy(predictions, test_y))
    return max(predictions_list), predictions_list.index(max(predictions_list)) + 1

def GBregressor(train_X,test_X,train_y_regression,test_y_regression):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    est = GradientBoostingRegressor(max_depth=5).fit(train_X,train_y_regression)
    train_X = train_X[train_X [:, 0].argsort()]
    plt.plot(sorted(train_X[:,0]),train_y_regression,alpha = 0.9,linewidth =2)
    plt.show()

def Ridgeregression(train_X,test_X,train_y_regression,test_y_regression):
    clf = Ridge()
    coefs = []
    scores = []
    alphas = np.logspace(-6, 6, 200)
    # Train the model with different regularisation strengths

    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(train_X, train_y_regression)
        coefs.append(clf.coef_)
        scores.append(clf.score(test_X,test_y_regression))

    y_predict = clf.predict(test_X)
    # Display results
    plt.figure(figsize=(20, 6))
    plt.subplot(131)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')

    plt.subplot(132)
    ax = plt.gca()
    ax.plot(alphas, scores)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Ridge score as a function of the regularization')
    plt.axis('tight')

    plt.subplot(133)
    ax = plt.gca()
    ax.scatter(y_predict,test_y_regression)
    plt.xlabel('y_predict')
    plt.ylabel('test_y_regression')

    plt.show()

def compare_result_classifier(train_X,test_X,train_y,test_y):
    Rf_a,Rf_p = random_forest(train_X,test_X,train_y,test_y)
    KNN_a,KNN_p = KNN(train_X, test_X, train_y, test_y)
    GB_a, GB_p = GBclassifer(train_X, test_X, train_y, test_y)
    model_names = ["Random Forest","KNN Classifier","GradientBoostingClassifier"]
    accuarcies = [Rf_a,KNN_a,GB_a]
    parameters = [Rf_p,KNN_p,GB_p]
    print(" |       Model       |     Accuracy    |     Parameter    ")
    print(" |-------------------|-----------------|----------------- ")
    for i in range(3):
        print(" | ", model_names[i], " | ",accuarcies[i]," | ",parameters[i])


if __name__ == "__main__":
    train_X,test_X,train_y,test_y,train_y_regression,test_y_regression = data_processing()
    compare_result_classifier(train_X,test_X,train_y,test_y)


    # GBregressor(train_X, test_X, train_y_regression, test_y_regression)
    # Ridgeregression(train_X, test_X, train_y_regression, test_y_regression)