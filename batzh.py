from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

###ensemble

# Create a list for the submodels: estimators
estimators = []

# Create a logistic regression model: model1
model1 = LogisticRegression(solver="newton-cg")

# Append model1 to the estimators list in format ('logistic', model1)
estimators.append(('logistic',model1))

# Create a decision tree model: model2
model2 = DecisionTreeClassifier()


# Append model1 to the estimators list in format ('decision_tree', model2)
estimators.append(("decision_tree",model2))

# Create an SVC model: model3
model3 = SVC(gamma='auto')

# Append model1 to the estimators list in format ('svm', model3)
estimators.append(("SVM",model3))


# Create the ensemble model from the estimators by VotingClassifier: ensemble 
ensemble = VotingClassifier(estimators,verbose=True)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

# Evaulate the model using the 10-fold cross validation: results
results = model_selection.cross_val_score(ensemble,X,Y,cv=10)

# Print the mean of the 10 results
print(results.mean())

# Check what result you get without cross-validation
# Fit your model on the training dataset and evaluate it
ensemble.fit(X_train,Y_train)
print(ensemble.score(X_test,Y_test))

###SVM

# Create a svm Classifier: clf
# Use linear kernel
from sklearn import svm
clf = svm.SVC(kernel='linear') 

# Train (fit) the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

###random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [2,10,200]
criterion = ['gini', 'entropy']
max_features =['auto']
max_depth = [5,10,15]
min_samples_split = [2,5,50]
min_samples_leaf = [1,4,20]
bootstrap = [True, False]

random_grid ={
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split':min_samples_split,
        'bootstrap': bootstrap
    
}

start_time = time.time()
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, 
                           param_distributions = random_grid,
                           n_iter = 10, 
                           cv = 10, 
                           random_state=42)
rf_random.fit(X_train,y_train)

print(f'Trains time: {time.time()-start_time}')

###knn

from sklearn.neighbors import KNeighborsClassifier


k = 3
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
knn.fit(X_train, y_train)


###metrics

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
