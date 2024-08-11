import pickle
import pandas as pd
from flask import Flask, render_template, request
import re
import nltk
from wordcloud import STOPWORDS
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer



modal = pickle.load(open("user_final_rating.pkl",'rb'))
tfidf_vectorizer = pickle.load(open('count_vector.pkl','rb'))
loaded_model = pickle.load(open("model.pkl",'rb'))
df = pd.read_csv("sample30.csv")


app = Flask(__name__)

# Example data for dropdown
options = modal.index.to_list()

stopword_list = set(STOPWORDS)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_option = request.form.get('dropdown')  # Get the selected option
        result = reccomend_products(selected_option)  # Generate a list based on the selection
        return render_template('index.html', options=options, result=result)

    return render_template('index.html', options=options, result=None)


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words


def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas


def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)


def reccomend_products(Username):
    """
    pass
    """
    new_df = pd.DataFrame()
    new_df['Product'] = modal.loc[Username].sort_values(ascending=False)[:20].reset_index()['name'].to_list()
    pos_perct = []
    for prod in new_df['Product']:
        prod_df = df[df['name']==prod]
        review_txt_ls = prod_df['reviews_text']
        review_lema_ls = review_txt_ls.apply(lambda text: normalize_and_lemmaize(text))
        review_tfidf_ls = tfidf_vectorizer.transform(review_lema_ls)
        count = 0
        for x in review_tfidf_ls:
            prob = loaded_model.predict(x)
            if prob[0] == 1:
                count += 1
        pos_perct.append(count/len(review_txt_ls))
    new_df['Percentage'] = pos_perct
    print(new_df.head())
    result = new_df.sort_values(by='Percentage', ascending=False)
    result = result['Product'].to_list()
    return result[:5]


def generate_list(option):
    # For simplicity, return a list of 5 elements based on the selected option
    return [f'{option} - Item {i+1}' for i in range(5)]


if __name__ == '__main__':
    app.run(debug=True)
