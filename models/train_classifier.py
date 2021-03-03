import sys

# import libraries
import pickle
import re

import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def load_data(database_filepath, table_name='message_and_categories'):
    '''
    Load data from database and returns X, Y and category names.
    
    Parameters
    --------------
    database_filepath: str
        file path of database
    table_name: str
        table name of database
    
    Returns
    ----------
    X: numpy.array
        features
    Y: numpy.array
        targets
    category_names: numpy.array
        names of targets
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM ' + table_name, engine)
    
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.columns[4:].values
    
    return X, y, category_names


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    '''
    Tokenize text
    
    Parameters
    ---------------
    text: str
        raw message
    lemmatizer: nltk.stem.WordNetLemmatizer
        lemmatizer
    
    Returns
    ----------
    tokens: list
        tokenized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detecte URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    '''
    Build the multi output classfire model
    
    Returns
    ----------
    cv: sklearn.model_selection.GridSearchCV
        model of multi output classifier
    '''
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'clf__estimator__learning_rate': [0.01, 0.1, 0.5],
        'clf__estimator__n_estimators': [10, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      scoring='f1_weighted', n_jobs=-1, verbose=3)

    return cv


def classification_scores(y_test, y_pred, category_names):
    '''
    return scores of accuracy, f1, precision and recall by DataFrame format.
    
    Parameters
    ---------------
    y_test: numpy.array
        true values
    y_pred: numpy.array
        predicted values
    category_names: list
        category names
    
    Returns
    ----------
    df_scores: pandas.DataFrame
        DataFrame of scores
    '''
    scores = []
    for i, category in enumerate(category_names):
        y_t, y_p = y_test[:, i], y_pred[:, i] 
        scores.append([category, 
                       accuracy_score(y_t, y_p), 
                       f1_score(y_t, y_p, average='weighted'),
                       precision_score(y_t, y_p, average='weighted'),
                       recall_score(y_t, y_p, average='weighted')])
    df_scores = pd.DataFrame(scores, columns=['category', 'accuracy', 'f1', 'precision', 'recall'])
    df_scores.set_index('category', inplace=True)
    return df_scores


def print_score_summary(df_scores, score_name):
    '''
    print out socres of min, max and mean.
    
    Parameters
    --------------
    df_scores: pandas.DataFrame
        socres dataframe
    socre_name: str
        score name of output ['accuracy', 'f1', 'precision', 'recall']
    '''
    print('[{}]'.format(score_name))
    print('mean : {:.2f}'.format(df_scores[score_name].mean()))
    print('min : {0:.2f} ({1})'.format(df_scores[score_name].min(), df_scores[score_name].idxmin()))
    print('max : {0:.2f} ({1})'.format(df_scores[score_name].max(), df_scores[score_name].idxmax()))
    print('mean : {:.2f}'.format(df_scores[score_name].mean()))
    print('')

    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Show evaluate of mode
    
    Parameters
    ---------------
    model: sklearn.model_selection.GridSearchCV
        classfier model
    X_test: numpy.array
        test features
    Y_test: numpy.array
        test target
    category_names: list
        category names
    '''
    # Predict categories of messages.
    Y_pred = model.predict(X_test)

    # print socre list
    df_scores = classification_scores(Y_test, Y_pred, category_names)
    print(df_scores)

    # print out summary
    print_score_summary(df_scores, 'accuracy')
    print_score_summary(df_scores, 'f1')
    print_score_summary(df_scores, 'precision')
    print_score_summary(df_scores, 'recall')


def save_model(model, model_filepath):
    '''
    Save trained model to pickle
    
    Parameters
    --------------
    model: sklearn.model_selection.GridSearchCV
        model to save
    model_filepath: str
        file path to save
    ''' 
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()