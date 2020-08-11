import  flask
import numpy as np

import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cleaned_df = pd.read_csv("cleaned_df.csv")
from sklearn.feature_extraction.text import TfidfVectorizer

all_titles = np.array(cleaned_df['title'])
tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
cleaned_df['overview'] = cleaned_df['overview'].fillna('')
tfv_matrix = tfv.fit_transform(cleaned_df['overview'])
from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
#we are using the sigmoid kernel to compute the similarity between two overviews
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(cleaned_df.index,index = cleaned_df['title']).drop_duplicates()

app = flask.Flask(__name__)
def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = int(indices[title])

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]
    tit = cleaned_df['title'].iloc[movie_indices]
    dat = cleaned_df['release_date'].iloc[movie_indices]

    # Top 10 most similar movies
    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df
# Set up the main route
@app.route('/')
def index():
    return flask.render_template('index.html')
@app.route('/recommend.html', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('recommend.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        #        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return (flask.render_template('negative.html', name=m_name))
        else:
            result_final = give_rec(m_name)
            names = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])

            return flask.render_template('positive.html', movie_names=names, movie_date=dates, search_name=m_name)


if __name__ == '__main__':
    app.run()