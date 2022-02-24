import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pandas as pd
from time import strftime

############# BASIC CLEAN ###################


def basic_clean(corpus):
    '''
    Basic text cleaning function  that  takes a corpus of text; lowercases everything; normalizes unicode characters; and replaces anything that is not a letter, number, whitespace or a single quote.
    '''
    lower_corpus = corpus.lower()
    normal_corpus = unicodedata.normalize('NFKD', lower_corpus)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
    basic_clean_corpus = re.sub(r"[^a-z0-9'\s]", '', normal_corpus)
    return(basic_clean_corpus)

############# BASIC CLEAN ###################

# Leave - in text


def basic_clean2(corpus):
    '''
    Basic text cleaning function  that  takes a corpus of text; lowercases everything; normalizes unicode characters; and replaces anything that is not a letter, number, whitespace or a single quote.
    '''
    lower_corpus = corpus.lower()
    normal_corpus = unicodedata.normalize('NFKD', lower_corpus)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
    basic_clean_corpus = re.sub(r"[^a-z0-9'\s]", ' ', normal_corpus)
    return(basic_clean_corpus)


############# BASIC CLEAN 3###################

def basic_clean3(corpus):
    '''
    Basic text cleaning function  that  takes a corpus of text; lowercases everything; normalizes unicode characters; and replaces anything that is not a letter, number, whitespace or a single quote.
    '''
    lower_corpus = corpus.lower()
    normal_corpus = unicodedata.normalize('NFKD', lower_corpus)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
    basic_clean_corpus = re.sub(r"[^-/A-Za-z0-9'\s]", ' ', normal_corpus)
    return(basic_clean_corpus)

##################### TOKEIZER ####################


def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return(tokenizer.tokenize(string, return_str=True))

####################### STEM #####################


def stem(text):
    '''
    Uses NLTK Porter stemmer object to return stems of words
    '''
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    stemmed_text = ' '.join(stems)
    return stemmed_text

################ LEMMATIZE ################


def lemmatize(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    lemmatized_text = ' '.join(lemmas)
    return(lemmatized_text)

################ REMOVE STOPWORDS #################


def remove_stopwords(string, extra_words=[], exclude_words=[]):
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


############### PREPARE ARTICLES ############


def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df, the name for a text column with the option to pass lists for extra_words and exclude_words and returns a df with the text article title, original text, stemmed text,lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    print("Renamed 'pn_history' column to 'original'")
    df['clean'] = df[column].apply(basic_clean3)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                   extra_words=extra_words,
                                   exclude_words=exclude_words)
    print('Added a basic clean column lowercaseing and removing special characters')
    df['stemmed'] = df[column].apply(basic_clean3)\
        .apply(tokenize)\
        .apply(stem)\
        .apply(remove_stopwords,
               extra_words=extra_words,
               exclude_words=exclude_words)
    print('Added stemmed column with tokenized words and stopwords removed')

    df['lemmatized'] = df[column].apply(basic_clean3)\
        .apply(tokenize)\
        .apply(lemmatize)\
        .apply(remove_stopwords,
               extra_words=extra_words,
               exclude_words=exclude_words)
    print('Added lemmatized column with lemmatized words and stopwords removed')
    print('Data preparation complete')
    return df[['id', 'case_num', 'pn_num', 'feature_num', 'feature_text', 'annotation', 'location', column, 'clean', 'stemmed', 'lemmatized']]


######## Prepare labeled data ########

def prep_train():
    # Load data
    df = pd.read_csv('train.csv')
    notes = pd.read_csv('patient_notes.csv')
    features = pd.read_csv('features_kaggle.csv')
    print('Test, notes, and features loaded.')
    # Merge dataframes
    df = df.merge(notes, how='inner', on='pn_num')
    df.drop(columns='case_num_y', inplace=True)
    df.rename(columns={'case_num_x': 'case_num'}, inplace=True)
    df = df.merge(features, how='inner', on='feature_num')
    df.drop(columns='case_num_y', inplace=True)
    df.rename(columns={'case_num_x': 'case_num'}, inplace=True)
    df.rename(columns={'pn_history': 'original'}, inplace=True)
    print('Merged dataframes')
    df = prep_article_data(
        df, 'original', extra_words=[], exclude_words=['no'])
    return df
