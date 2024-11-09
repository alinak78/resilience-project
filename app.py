from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def topic_modeling(descriptions):
    # Generate bigrams
    bigrams = [' '.join(gram) for desc in descriptions for gram in ngrams(preprocess_text(desc), 2)]
    
    # Vectorize text for LDA
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(descriptions)
    
    # Sklearn LDA
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)
    
    # Get topics for Sklearn
    sklearn_topics = []
    num_words = 10
    for topic in lda_model.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        sklearn_topics.append(top_words)
    
    # Gensim LDA
    texts = [preprocess_text(description) for description in descriptions]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_gensim = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42)
    
    # PyLDAvis HTML
    lda_display = pyLDAvis.gensim.prepare(lda_gensim, corpus, dictionary)
    lda_html = pyLDAvis.prepared_data_to_html(lda_display)

    return bigrams[:10], sklearn_topics, lda_html

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    df = pd.read_csv(file)
    descriptions = df['Description'].dropna().tolist()
    
    bigrams, sklearn_topics, lda_html = topic_modeling(descriptions)
    
    return jsonify({
        "bigrams": bigrams,
        "sklearn_topics": sklearn_topics,
        "lda_html": lda_html
    })

if __name__ == "__main__":
    app.run(debug=True)
