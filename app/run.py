import json
import pickle
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_and_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # categories usage rates
    cat_titles = df.columns[4:].values
    cat_rates = []
    for title in cat_titles:
        cat_rates.append([title, df[title].sum()  / df.shape[0]])
    df_cat_rates = pd.DataFrame(cat_rates, columns=['cat', 'rate'])
    df_cat_rates = df_cat_rates.sort_values('rate', ascending=False).reset_index(drop=True)
    
    cat_rates = df_cat_rates['rate'].values
    categories=df_cat_rates['cat'].values

    # Kinds number of message category
    cat_kinds = df.iloc[:, 4:].sum(axis=1)
    kinds_num = [str(x) for x in range(0, 15)]
    kinds_num.append('15+')
    kinds_counts = []
    for i in range(0, 15):
        kinds_counts.append(len(cat_kinds[cat_kinds == i]))
    kinds_counts.append(len(cat_kinds[cat_kinds > 14]))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories,
                    y=cat_rates
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Rate"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=kinds_num,
                    y=kinds_counts
                )
            ],

            'layout': {
                'title': 'Kinds number in one message',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Kinds"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()