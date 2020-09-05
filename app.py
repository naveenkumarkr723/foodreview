from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


# load the model from disk

clf = joblib.load('foodreview_pkl')
cv = joblib.load('trans_count_vect')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        
        predictions = clf.predict(vect)
    return render_template('result.html', prediction = predictions)



if __name__ == '__main__':
    app.run(debug=True)
