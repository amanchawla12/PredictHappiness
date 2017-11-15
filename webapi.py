import pandas as pd
from flask import Flask, render_template, request
from nltk import PorterStemmer, re
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from script import cleanData

app = Flask(__name__)
clf = joblib.load('classifier.pkl')
tfidfvec = joblib.load("tfidf.pkl")



@app.route("/feedback")
def helloWorld():
    return render_template("feedback.html")

@app.route("/process" , methods=['POST'])
def process():
    if request.method == 'POST':
        browser = request.form['browser']
		#print(browser)
        device = request.form['device']
        description = request.form['description']
        feat = to_feat(description, device, browser)
        #clf = joblib.load('classifier.pkl')
        pred = clf.predict(feat)
		
        if pred==0:
            pred="Not Happy"
        else :
            pred="Happy"
        print(pred)
        return render_template('response.html',browser=browser , device=device , description=description, pred=pred)

def cleanData(text):
    stops = set(pd.read_csv("Data/stopwords.csv", header=None)[0])
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]', r'', txt)
    txt = re.sub(r'\n', r' ', txt)

    txt = " ".join([w.lower() for w in txt.split()])

    st = PorterStemmer()
    txt = " ".join([st.stem(w) for w in txt.split()])

    txt = " ".join([w for w in txt.split() if w not in stops])

    return txt

def to_feat(description, device, browser):
    txt = cleanData(str(description))
    #print(txt)
    #tfidfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=1000)
    #tfidfvec = joblib.load("tfidf.pkl")
    fr=pd.DataFrame({'Description' : [txt]})
    #print(fr.head())
    tfidfdata = tfidfvec.transform(fr['Description'])
    tfidf_df = pd.DataFrame(tfidfdata.todense())

    tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]

    df = pd.read_csv("Data/browsers.csv")

    num1 = df.index[df['Browsers']==browser][0]
    num2=0
    if device=='Mobile':
        num2=1
    elif device=='Tablet':
        num2=3

    df2 = pd.DataFrame({'Browser_Used': [num1], 'Device_Used': [num2]})

    final_df = pd.concat([df2, tfidf_df], axis=1)
    #print(tfidfvec.vocabulary_)
    #print(list(final_df.iloc[0]))
    #print(description)
    return final_df

#def init():
#    clf = joblib.load('classifier.pkl')

#init()
app.run()