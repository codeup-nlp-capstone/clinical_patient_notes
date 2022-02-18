import unicodedata
import re
import json
import os
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
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
    cleaned = re.sub(r"[^a-z0-9'\s]", '', lower)
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
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df[['category', column, 'stemmed', 'lemmatized']]
        
'''========================================================================'''