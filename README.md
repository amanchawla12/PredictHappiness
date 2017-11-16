# PredictHappiness
Text Classification using SVM
  
**Dependencies required:**
- Python 3.6
- Sklearn
- flask
- pandas
- nltk
- re
- pickle

**About the data:**
The Feature included text, Device, Browser.

"raw_train.csv" contains the raw text data and "new_train.csv" contains the cleaned data(stemmed).

The model is serialized using pickle library and the model is exposed using API written using flask framework. For testing purpose the input
is taken from the feedback.html and output is shown on response.html


**Run the project:**
- open the terminal and pip all the dependencies
- run command "python webapi.py" in project folder
- open 127.0.0.1:5000/feedback in browser

**About the files**
1. *.pkl     - serialized files
2. script.py - text pre-processing and building the model
3. webapi.py - backend of form using flask
