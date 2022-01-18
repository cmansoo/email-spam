import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

df = pd.read_csv('clean.csv')

# train-test split, training set will be 80% of the data, random_state is seed for randomization
train = df.sample(frac=0.8, random_state=1)
test = df.drop(train.index)

# feature extraction
count_vect = CountVectorizer()
features = count_vect.fit_transform(train['clean_text'])

# predictive model using support vector machine
model_s = svm.SVC()
model_s.fit(features, train['spam'])

# test accuracy
features_test = count_vect.transform(test['clean_text'])
print(f'Accuracy of SVM: {model_s.score(features_test, test["spam"])}')

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, train['spam'])

# test accuracy
print(f'Accuracy of KNN: {knn.score(features_test, test["spam"])}')
