import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
df = pd.read_csv('./spam.csv',delimiter="\t", error_bad_lines=False)
df["Label"]=df.index
df.index = np.arange(1, len(df) + 1)
x=df.iloc[:,0]
y=df.iloc[:,1]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(x,y)

print("Sample printing : ",X_train_raw.iloc[0],"\n")


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i in range(0,len(y_test)):
    print(i,"Original :",y_test.iloc[i])
    print("\tpredicted :",predictions[i])
    

