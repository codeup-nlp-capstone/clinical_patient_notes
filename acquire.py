import unicodedata
import re
import json
import os

from requests import get
from sklearn.model_selection import train_test_split

# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.corpus import stopwords
from time import strftime


# This function will create a dictionary of the wanted data.
def parse_person(person):
    name = person.h2.text
    quote = person.p.text.strip()
    email = person.select('.email')[0].text
    phone = person.select('.phone')[0].text
    address = [l.strip() for l in person.select('p')[-1].text.split('\n')[1:3]]


    return {
        'name': name, 'quote': quote, 'email': email,
        'phone': phone,
        'address': address
    }


# This function takes in a url as an argument and parse the contents.
def parse_blog(url):
    url = url.get('href')
    response = get(url, headers={'user-agent': 'Codeup DS Hopper'})
    blog = BeautifulSoup(response.text)
    title = blog.h1.text
    date_source = blog.p.text
    content = blog.find_all('div',class_ = 'entry-content')[0].text

    return {
        'title': title, 'date & source': date_source, 'original': content
    }

# This function will loop through a list of urls and return a dataframe.
def get_codeup_blogs(cached=False):
    if cached == True:
        df = pd.read_json('codeup_blogs.json')
        return df
    else:
        # Fetch data
        response = get('https://codeup.com/blog/', headers={'user-agent':'Codeup DS Hopper'})
        soup = BeautifulSoup(response.text)
        urls = soup.find_all('a',class_ = 'more-link')

        blog_df = pd.DataFrame([parse_blog(url) for url in urls])
        # save the dataframe as json:
        blog_df.to_json('codeup_blogs.json')

        return blog_df

# This function will return the text from an online article.
def get_article_text():
    # Read data locally if it exists.
    if os.path.exists('article.txt'):
        with open('article.txt') as f:
            return f.read()
    # Fetch data
    url = 'https://codeup.com/data-science/math-in-data-science/'
    headers = {'User-Agent': 'Codeup Data Science'}
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.text)
    article = soup.find('div', id='main-content')

    # Save it for when needed
    with open('article.txt', 'w') as f:
        f.write(article.text)

    return article.text


'''========================================================================'''
def parse_news_card(card):
    'Given a news card object, returns a dictionary of the relevant information.'
    card_title = card.select_one('.news-card-title')
    output = {}
    output['title'] = card.find('span', itemprop = 'headline').text
    output['author'] = card.find('span', class_ = 'author').text
    output['original'] = card.find('div', itemprop = 'articleBody').text
    output['date'] = card.find('span', clas ='date').text
    return output


def parse_inshorts_page(url):
    '''Given a url, returns a dataframe where each row is a news article from the url.
    Infers the category from the last section of the url.'''
    category = url.split('/')[-1]
    response = requests.get(url, headers={'user-agent': 'Codeup DS'})
    soup = BeautifulSoup(response.text, 'lxml')
    cards = soup.select('.news-card')
    df = pd.DataFrame([parse_news_card(card) for card in cards])
    df['category'] = category
    return df

def get_inshorts_articles():
    '''
    Returns a dataframe of news articles from the business, sports, technology, and entertainment sections of
    inshorts.
    '''
    today = strftime('%Y-%m-%d')
    if os.path.isfile(f'inshorts-{today}.json'):
        # If json file exists, read in data from json file.
        df = pd.read_json(f'inshorts-{today}.json')
        return df
    else:
        url = 'https://inshorts.com/en/read/'
        categories = ['science', 'business', 'sports', 'technology', 'entertainment']
        df = pd.DataFrame()
        for cat in categories:
            df = pd.concat([df, pd.DataFrame(parse_inshorts_page(url + cat))])
        df = df.reset_index(drop=True)
        # save the dataframe as json:
        df.to_json(f'inshorts-{today}.json')
        return df

'''========================================================================'''
def basic_clean(string):
    # Lowercase everything in the text.
    lower = string.lower()
    lower = unicodedata.normalize('NFKD', lower)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    # Remove anything that isn't a-z, a number, single quote, or whitespace.

    # cleaned = re.sub(r"[^a-z0-9'\s]", '', lower)
    cleaned = re.sub(r"[\W]", ' ', lower)
    return cleaned

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)

def stem(string):
    # Create the nltk stemmer object, then use it
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(string, extra_words =[], exclude_words =[]):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list.
    stopword_list = stopwords.words('english')

    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)

    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()

    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]

    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)

    return string_without_stopwords


# This function will take in a dataframe of news/blog articles and prepare the text in three different ways.
def prep_text(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                    extra_words=extra_words,
                                    exclude_words=exclude_words)

    df['stemmed'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(stem)\
                            .apply(remove_stopwords,
                                    extra_words=extra_words,
                                    exclude_words=exclude_words)

    df['lemmatized'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(lemmatize)\
                            .apply(remove_stopwords,
                                    xtra_words=extra_words,
                                    exclude_words=exclude_words)

    return df[['case', column, 'stemmed', 'lemmatized']]

'''========================================================================'''

def analyze_text(string):
    # Get length of total characters in all cleaned science articles.
    total_characters = len(string)
    print(f'Total amount of characters: {total_characters}')

    # Get wordcount of all words in cleaned science articles.
    total_words = len(string.split())
    print(f'Total amount of words: {total_words}')

    # Get list of unique words and a count in cleaned science articles.
    unique_words = pd.DataFrame(string.split())[0].unique()
    print('Total amount of unique words: ',len(unique_words))

    # Get average word length of all words in cleaned science articles.
    avg_wordlength = round(pd.Series([len(word) for word in unique_words]).mean(), 1)
    print('Average word length: ', avg_wordlength)

    # Get the ratio of unique words
    unique_ratio = len(unique_words) / (total_words)
    print('The ratio of unique words: ', unique_ratio)

    # Get length of every unique word and plot a histogram of how many times each length of word appears.
    list_of_graph_titles = news_df.category.unique()
    plt.figure(figsize=(10,8))
    sns.histplot([len(word) for word in unique_words], binwidth=1)
    plt.xlabel('character_count')
    plt.title('Number of Characters in Each Word')

'''========================================================================'''

def boil_it_down(df, column):
    cleaned_column = df[column].apply(basic_clean)
    lists_of_targets = []
    for target in cleaned_column:
        lists_of_targets.append(list(re.split(r'\bor', target)))
    list_of_targets = []
    for ailments in lists_of_targets:
        for ailment in ailments:
            list_of_targets.append(ailment)
    list_of_targets = [s.strip() for s in list_of_targets]
    return list_of_targets

'''========================================================================'''

def prep_text2(df, column, extra_words=[], exclude_words=['no','i']):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    return df[['case', column, 'clean']]

'''========================================================================'''

def split_data(df):
    test_split = 0.1

    # Initial train and test split.
    train_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df['case'].values,
    )

    # Splitting the test set further into validation and new test set.
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")
    return train_df, val_df, test_df

'''========================================================================'''

def prep_and_split_data():
    
    # Read csv files into a Pandas dataframe.
    features = pd.read_csv('features.csv')
    notes = pd.read_csv('patient_notes.csv')
    # Rename columns in the features dataframe.
    features.rename(columns={'feature_num':'feature_id', 'case_num':'case', 'feature_text':'target'}, inplace=True)
    # Rename columns in notes dataframe.
    notes.rename(columns={'pn_num':'note_id', 'case_num':'case', 'pn_history':'student_notes'}, inplace=True)
    # Clean 'student_notes'
    notes = prep_text2(notes, 'student_notes')
    # Run text through 'basic_clean' function.
    features['cleaned_targets'] = features.target.apply(basic_clean)
    list_of_targets = boil_it_down(features, 'target')
    # Create a list of targets for each case
    case_0_targets = list(features[features.case == 0].cleaned_targets)
    case_1_targets = list(features[features.case == 1].cleaned_targets)
    case_2_targets = list(features[features.case == 2].cleaned_targets)
    case_3_targets = list(features[features.case == 3].cleaned_targets)
    case_4_targets = list(features[features.case == 4].cleaned_targets)
    case_5_targets = list(features[features.case == 5].cleaned_targets)
    case_6_targets = list(features[features.case == 6].cleaned_targets)
    case_7_targets = list(features[features.case == 7].cleaned_targets)
    case_8_targets = list(features[features.case == 8].cleaned_targets)
    case_9_targets = list(features[features.case == 9].cleaned_targets)
    case_targets = pd.DataFrame({'case':[n for n in np.arange(10)], 'targets':[case_0_targets,case_1_targets,case_2_targets,case_3_targets,case_4_targets,case_5_targets,case_6_targets,case_7_targets,case_8_targets,case_9_targets]})
    df = notes.merge(case_targets, how='inner', on='case')
    train_df, val_df, test_df = split_data(df)
    return train_df, val_df, test_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    