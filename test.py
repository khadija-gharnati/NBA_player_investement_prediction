import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

def score_classifier(dataset,classifier,labels, cv, score_function = f1_score):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=cv,random_state=50,shuffle=True)
    #confusion_mat = np.zeros((2,2))
    score = 0
    recall = 0
    precision = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        score += score_function(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
    score/=cv
    recall/=cv
    precision/= cv


    return score, recall, precision
# Load dataset
df = pd.read_csv(".\\nba_logreg.csv", sep = ";")
df = df.fillna(0.0)
# extract names, labels, features names and values
names = df['Name'].values.tolist() # players names
labels = df['TARGET_5Yrs'].values # labels
paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values

# replacing Nan values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x]=0.0


#liste de features séléctionnées
final_features = ['GP', 'MIN', 'PTS', 'FTM', 'OREB', 'DREB', 'TOV', 'STL', 'FG%', 'BLK']
# normalize dataset
X = MinMaxScaler().fit_transform(df[final_features])
y = df['TARGET_5Yrs'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# les paramétres des modèles à utiliser dans GridsearchCV
params_LR = {
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter' :[100, 300, 500, 600, 750, 900, 1000]
}
params_KNN = {
    'n_neighbors' : [1, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
}
params_RF = {
    'n_estimators' : [1000, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_features' : ['sqrt', 'log2', None, 'auto'],
    'max_depth' : [None, 1, 2, 3, 4, 5, 6]
}
params_GB = {
    'loss' : ['log_loss', 'exponential'],
    'learning_rate':[0.05, 0.075, 0.1],
    'criterion' : ['friedman_mse', 'squared_error', 'squared_error']
}
params_SVC = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma':['scale', 'auto']
}

# modèles à tester
models_Selected_feats = [
    GridSearchCV(estimator=LogisticRegression(), param_grid=params_LR, n_jobs=-1, cv=5, verbose=0).fit(X_train, y_train).best_estimator_,
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params_KNN, n_jobs=-1, cv=5, verbose=0).fit(X_train, y_train).best_estimator_,
   #GridSearchCV(estimator=RandomForestClassifier(), param_grid=params_RF, n_jobs=-1, cv=5).fit(X_train, y_train).best_estimator_,
    GridSearchCV(estimator=GradientBoostingClassifier(),  param_grid=params_GB, n_jobs=-1, cv=5, verbose=0).fit(X_train, y_train).best_estimator_,
    GridSearchCV(estimator=SVC(), param_grid=params_SVC, n_jobs=-1, cv=5, verbose=0).fit(X_train, y_train).best_estimator_ ,
    XGBClassifier(use_label_encoder=False),
    GaussianNB(),
    AdaBoostClassifier()
]

results  = pd.DataFrame(columns = ['model_name', 'f1_score', 'recall', 'precision'])
for model in models_Selected_feats:

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_mat = np.zeros((2, 2))
    confusion_mat[0] = np.array(confusion_matrix(y_pred, y_test))[:, 0]/(y_test.shape[0] - sum(y_test))
    confusion_mat[1] = np.array(confusion_matrix(y_pred, y_test))[:, 1]/sum(y_test)

    score_result, recall, precision = score_classifier(X,model,labels, cv = 15)
    entry = [model.__class__.__name__, score_result, recall, precision]
    results.loc[len(results.index)] = entry

results = results.sort_values(by = 'f1_score', ascending=False)
print(results)

#le modèle choisi
model = models_Selected_feats[0]
print('le modèle choisi est : ', model)


# Pipeline
model_pipeline = make_pipeline(MinMaxScaler(), model)
final_features = ['GP', 'MIN', 'PTS', 'FTM', 'OREB', 'DREB', 'TOV', 'STL', 'FG%', 'BLK']
Data = df[final_features].values
Labels = df['TARGET_5Yrs'].values


#entainer le modèle choisi
X_train, X_test, y_train, y_test = train_test_split(Data, Labels, train_size=0.2, random_state=50)
model_pipeline.fit(X_train, y_train)

# évaluer le modèle
y_pred = model_pipeline.predict(X_test)
score = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print(f'le meilleur score obtenue = {score}')
print(f'la matrice de confusion = {confusion_mat}')

import joblib
joblib.dump(model_pipeline, 'Prediction_model')
print('le modèle a été enregistré avec succés')


