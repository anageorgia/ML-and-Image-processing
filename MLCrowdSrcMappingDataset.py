import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import requests
import time



start_time = time.time()

print('\n - Lendo o arquivo com o dataset')
data = pd.read_csv('training.csv')
data_app = pd.read_csv('testing.csv')


# Floresta, fazenda, inacessível, água, grama, arbusto
class_col_numerical = {"class": {"forest": 0, "farm": 1, "impervious": 2, "water": 3, "grass": 4, "orchard": 5}}

# Renomeia itens da coluna class de nomes para números
data.replace(class_col_numerical, inplace=True)
data_app.replace(class_col_numerical, inplace=True)

# Renomeia a coluna class para outcome
data.rename(columns = {'class':'outcome'}, inplace = True)
data_app.rename(columns = {'class':'outcome'}, inplace = True)


print('\n - Criando X e y para o algoritmo de aprendizagem a partir do arquivo training.csv')

feature_cols = list(data)

X = data[feature_cols]

y = data.outcome

# Separa os dados entre treinamento e teste, utilizando validação cruzada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)




# Criando um classificado árvore de decisão
print('\n - Criando modelo preditivo Decision Tree Classifier')

# ----------------------
# training a DescisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

accuracy_dtree = accuracy_score(y_test, dtree_predictions)

print('\n CLASSIFICADOR DECISION TREE')
print('\n - Matriz de confusão: \n\n', cm)
print('\n - Acurácia: ', accuracy_dtree*100)




# ----------------------
# training a linear SVM classifier
print('\n\n ->>>  |Criando modelo preditivo SVM|  <<<-\n')
from sklearn.svm import SVC

#svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_model_linear = SVC(kernel='linear').fit(X_train, y_train)

# Description:
# C : float, optional (default=1.0) - Penalty parameter C of the error term.
# kernel: string, optional (default=’rbf’) - Specifies the kernel type to be used in the algorithm. (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable)
# tol: float, optional (default=1e-3) - Tolerance for stopping criterion.

svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

print('\n - Matriz de confusão:\n')
print(cm)

accuracy_svm = accuracy_score(y_test, svm_predictions)

print('\n\n - Acurácia SVM:\n')
print(accuracy_svm)



# -----------------------
print('\n - Criando modelo preditivo KNN')
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

# accuracy on X_test
accuracy = knn.score(X_test, y_test)

# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)

print('\n CLASSIFICADOR KNN')
print('\n - Matriz de confusão: \n\n', cm)
print('\n - Acurácia: ', accuracy*100)



# -----------------------
print('\n - Criando modelo preditivo Naive Bayes')
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)

# accuracy on X_test
accuracy = gnb.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)

print('\n CLASSIFICADOR NAIVE BAYES')
print('\n - Matriz de confusão: \n\n', cm)

print('\n - Acurácia:', accuracy*100)


#-------------------
print("\n - Tempo de execução: %s segundos" % (time.time() - start_time))
