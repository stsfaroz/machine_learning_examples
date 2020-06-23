import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
def tokens(x):
    return x.split(',')
tfidf_vect= TfidfVectorizer( tokenizer=tokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)

pipeline = Pipeline([
 ('vect', TfidfVectorizer(stop_words='english')),
 ('clf', LogisticRegression())
])

parameters = {
 'vect__max_df': (0.25, 0.5, 0.75),
 'vect__stop_words': ('english', None),
 'vect__max_features': (2500, 5000, 10000, None),
 'vect__ngram_range': ((1, 1), (1, 2)),
 'vect__use_idf': (True, False),
 'vect__norm': ('l1', 'l2'),
 'clf__penalty': ('l2', 'l2'),
 'clf__C': (0.01, 0.1, 1, 10),
}


if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy', cv=3)
    df = pd.read_csv('./spam.csv',delimiter="\t", error_bad_lines=False)
    df["Label"]=df.index
    df.index = np.arange(1, len(df) + 1)
    x=df.iloc[:,0]
    y=df.iloc[:,1]
    y=le.fit_transform(y)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(x,y)
    grid_search.fit(X_train_raw, y_train)
    print ('Best score: %0.3f' % grid_search.best_score_)
    print ('Best parameters set:' )
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test_raw)
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('Precision:', precision_score(y_test, predictions))
    print ('Recall:', recall_score(y_test, predictions))
