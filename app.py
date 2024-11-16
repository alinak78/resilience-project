import pandas as pd
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
import json

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load the CSV file
df = pd.read_csv("climate_adaptation_solutions-2.csv")
descriptions = df['Description'].dropna().tolist()

# Preprocess text function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Generate bigrams function
def generate_bigrams(texts):
    bigrams = [' '.join(gram) for desc in texts for gram in ngrams(preprocess_text(desc), 2)]
    return bigrams

# Generate bigrams for descriptions
bigrams = generate_bigrams(descriptions)
print("Sample bigrams:", bigrams[:10])  # Display sample bigrams

# Sklearn LDA for topic modeling
def sklearn_lda(dtm, vectorizer, num_topics=10, num_words=12):
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)
    
    topics = []
    for topic in lda_model.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        topics.append(top_words)
    return topics

# Vectorize text and apply LDA with sklearn
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(descriptions)
sklearn_topics = sklearn_lda(dtm, vectorizer)
print("Sklearn Topics:", sklearn_topics)

# Gensim LDA for topic modeling
def gensim_lda(texts, num_topics=10, passes=15):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)
    return lda_model, corpus, dictionary

# Preprocess text for Gensim LDA
texts = [preprocess_text(description) for description in descriptions]
lda_gensim, corpus, dictionary = gensim_lda(texts)

# Save Gensim LDA visualization as HTML
lda_display = pyLDAvis.gensim.prepare(lda_gensim, corpus, dictionary)
pyLDAvis.save_html(lda_display, "lda_visualization_english.html")
print("LDA visualization saved as 'lda_visualization_english.html'")



