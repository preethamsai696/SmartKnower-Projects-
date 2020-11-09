import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, render_template, request,jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/trap', methods=['POST'])
def index():
    if request.method == 'POST':
        SearchString = request.form['content']
        try:
            l = []
            df = pd.read_csv('imdbset.csv')
            x = df.iloc[:, 0].values
            y = df['sentiment'].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            model = Pipeline([('tfidf', TfidfVectorizer()), ('model', MultinomialNB())])
            model.fit(x_train, y_train)
            y_pred = model.predict([SearchString])
            l.append(y_pred)
            return render_template('results.html',l=l)
        except:
            return 'Something is wrong'

if __name__ == "__main__":
    app.run(port=8000,debug=True)






