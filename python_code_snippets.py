# Decision Tree visualization using pydot

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 

# ------------------------------------------------------------------------------------------------------------------------
# elbow method for KNN

error_rate = []

for i in range(1,40):           
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')   

# ------------------------------------------------------------------------------------------------------------------------
## tfidf vectors and their weights

# TF = (Number of time the word occurs in the text) / (Total number of words in text)
# IDF = (Total number of documents / Number of documents with word t in it)
# TF-IDF = TF * IDF

# creating BOW tokens from input text
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(sentences)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

# sort words by their weights of tf-idf transformer
weights_df.sort_values(by='weight', ascending=False).head(10)


# ------------------------------------------------------------------------------------------------------------------------

# Bayesian Parameter tuning using hyper opt

import time
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
# the search space
space={'max_depth': hp.choice('max_depth', [3,4,5,6]),
        'gamma': hp.choice('gamma', [i for i in range(1,10)]),
        'colsample_bytree' : hp.choice('colsample_bytree', [i/10 for i in range(1,10)]),
        'learning_rate' : hp.choice('learning_rate', [.001,.005,.01,.05,.1]),
        'n_estimators' : hp.choice('n_estimators', [i for i in range(100,1000)]),
        'subsample': hp.choice('subsample', [.1,.2,.3,.4,.5]),
        'seed': 6
    }
# search algorithm
def objective(space):
    clf=xgboost.XGBRegressor(
                    n_estimators =space['n_estimators'], max_depth = space['max_depth'], gamma = space['gamma'],
                    colsample_bytree = space['colsample_bytree'],learning_rate=space['learning_rate'],
                    subsample=space['subsample'])
    
    evaluation = [( x_train, y_train), ( x_test, y_test)]
    
    clf.fit(x_train, y_train,
            eval_set=evaluation, eval_metric="mae",
            early_stopping_rounds=20,verbose=False)
    

    pred = clf.predict(x_test)
    accuracy = mean_absolute_error(y_test,pred)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()
# finding the best combination of hyperparameters by minimizing the onjective function defined above on the defined loss fn and search space
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 50,
                        trials = trials)


print("The best hyperparameters are : ","\n")
print(best_hyperparams)