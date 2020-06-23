import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = pd.read_csv('./spam.csv',delimiter="\t", error_bad_lines=False)
df["Label"]=df.index
df.index = np.arange(1, len(df) + 1)
x=df.iloc[:,0]
y=df.iloc[:,1]
y=le.fit_transform(y)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(x,y)

#X_train_raw=x
#y_train=y
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predict = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, predict, pos_label=2)
roc_auc = auc(y_test, predict)
# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
