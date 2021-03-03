import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load Disaster Responce Data
    
    Parameters
    ---------------
    messages_filepath: str
        file path of disaster messages
    categories_filepath: str
        file path of categories
    
    Returns
    ----------
    df: pandas.DataFrame
        messages and categories mereged
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories)
    return df

def clean_data(df):
    '''
    Clean data for machine learning.
    
    Parameters
    df: pandas.DataFrame
       raw data of messages and categories just merged
    
    Returns
    ----------
    df: pandas.DataFrame
        cleand data
    '''
    # categories splitted by ';'
    cat = df['categories'].str.split(';', expand=True)
    # columns of categories
    cols = cat.iloc[0].str.split('-', expand=True).iloc[:,0].values
    # set columns
    cat.columns = cols
    # use only trailing number 0 or 1
    for label in cols:
        cat[label] = cat[label].str[-1]
    # convert to int
    cat = cat.astype(int)
    # there are numbers other than 0/1 so convert it.
    cat = (cat > 0).astype(int) # False to 0, True to 1
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, cat], axis=1)
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df
    

def save_data(df, database_filename, table_name='message_and_categories'):
    '''
    Save DataFrame to Database
    
    Parameters
    ---------------
    df: pandas.DataFrame
        DataFrame to save
    database_filename: str
        File name of DataBase
    table_name: str
        Table name of DataBase, default is 'message_and_categories'
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()