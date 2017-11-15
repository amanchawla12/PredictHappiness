import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("Data/new_train.csv")
test = pd.read_csv("Data/new_test.csv")
stops = set(pd.read_csv("Data/stopwords.csv", header=None)[0])

print("reading done")
"""
def cleanData(text):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]', r'', txt)
    txt = re.sub(r'\n', r' ', txt)

    txt = " ".join([w.lower() for w in txt.split()])

    st = PorterStemmer()
    txt = " ".join([st.stem(w) for w in txt.split()])

    txt = " ".join([w for w in txt.split() if w not in stops])

    return txt
"""

test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)

#alldata['Description'] = alldata['Description'].map(lambda x: cleanData(x))

#print("replacing of browsers started...")
alldata.replace({'Mozilla Firefox': 'Firefox' , 'Mozilla': 'Firefox' , 'Google Chrome':'Chrome', 'Internet Explorer':'IE',
                 'InternetExplorer':'IE'}, regex=True, inplace=True)

#print(alldata['Browser_Used'].value_counts())

print("Cleaning of data is DONE")

tfidfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=1000)

tfidfdata = tfidfvec.fit_transform(alldata['Description'])
joblib.dump(tfidfvec, "tfidf.pkl")

print("Making features of data is DONE")

cols = ['Browser_Used', 'Device_Used']

for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x])
    if x=="Browser_Used":
        df = pd.DataFrame({'Browsers':lbl.classes_})
        df.to_csv("Data/browsers.csv", index=False)


tfidf_df = pd.DataFrame(tfidfdata.todense())

print("features done")

tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]


tfid_df_train = tfidf_df[:len(train)]
tfid_df_test = tfidf_df[len(train):]

train_feats1 = alldata[~pd.isnull(alldata.Is_Response)]
test_feats1 = alldata[pd.isnull(alldata.Is_Response)]

train_feats1['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats1['Is_Response']]

target = train_feats1['Is_Response']

train_feats=tfid_df_train
test_feats=tfid_df_test
train_feats = pd.concat([train_feats1[cols], tfid_df_train], axis=1)
test_feats = pd.concat([test_feats1[cols], tfid_df_test], axis=1)

print("PreProcessing on data is Done")


#From here on we will work on classifier


#print("Fitting data for tf-idf representation")

clf = LinearSVC()
print("fitting the data in svm")
clf.fit(train_feats, target)
joblib.dump(clf, "classifier.pkl")
#clf=joblib.load('classifier.pkl')
#prediction of actual data starts

print("making predictions of actual test data of tf-idf Representation")
pred = clf.predict(test_feats)
#print(list(test_feats.iloc[0]))

def to_labels(x):
    if x==1:
        return "happy"
    return "not_happy"

sub = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':pred})
sub['Is_Response'] = sub['Is_Response'].map(lambda x:to_labels(x))


sub = sub[['User_ID', 'Is_Response']]

print("writing data into sub.csv")
sub.to_csv("Data/sub1.csv", index=False)

print("success")