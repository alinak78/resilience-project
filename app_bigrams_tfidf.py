import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud
from io import BytesIO
import base64

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.util import ngrams

# Load the CSV file
df = pd.read_csv("climate_adaptation_solutions-2.csv")

# Use only the 'Description' column
if 'Description' in df.columns:
    descriptions = df['Description'].dropna().tolist()
else:
    raise ValueError("'Description' column is missing in the dataset.")

# Preprocess text function (noun-noun bigrams only, stricter filtering)
def preprocess_text_with_noun_noun_bigrams(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Extract noun-noun bigrams and filter duplicates
    bigrams = [
        f"{word1} {word2}"
        for (word1, tag1), (word2, tag2) in ngrams(pos_tags, 2)
        if tag1.startswith('NN') and tag2.startswith('NN') and word1 != word2
    ]
    return bigrams

# Preprocess all descriptions to noun-noun bigrams
texts = [preprocess_text_with_noun_noun_bigrams(text) for text in descriptions]

# Flatten bigrams for TF-IDF processing
flat_bigrams = [" ".join(text) for text in texts if text]

# Normalize bigrams by removing spaces and filtering duplicates
def normalize_and_filter_bigrams(bigrams):
    normalized = set()
    filtered_bigrams = []
    for bigram in bigrams:
        # Split and ensure the bigram has exactly two words
        words = bigram.split()
        if len(words) == 2:
            word1, word2 = words
            # Normalize by sorting words alphabetically and removing duplicates
            normalized_form = "".join(sorted([word1, word2]))
            if normalized_form not in normalized:
                normalized.add(normalized_form)
                filtered_bigrams.append(bigram)
    return filtered_bigrams


filtered_bigrams = normalize_and_filter_bigrams(flat_bigrams)

# Apply TF-IDF
# Apply TF-IDF with adjusted parameters
vectorizer = TfidfVectorizer(
    max_df=0.99,  # Allow terms in up to 99% of documents
    min_df=1,     # Include terms appearing in at least 1 document
    stop_words='english', 
    ngram_range=(2, 2)  # Bigram processing
)

tfidf_matrix = vectorizer.fit_transform(filtered_bigrams)
tfidf_features = vectorizer.get_feature_names_out()


# Convert TF-IDF scores to tokens for Gensim LDA
tfidf_tokens = [
    [tfidf_features[idx] for idx in tfidf_matrix[row].indices]
    for row in range(tfidf_matrix.shape[0])
]

# Gensim LDA for topic modeling (with TF-IDF transformed bigrams)
def gensim_lda_with_tfidf(tokens, num_topics=10, passes=15):
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)
    return lda_model, corpus, dictionary

lda_gensim, corpus, dictionary = gensim_lda_with_tfidf(tfidf_tokens)

# Save Gensim LDA visualization as HTML
lda_display = pyLDAvis.gensim.prepare(lda_gensim, corpus, dictionary)
pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_display)

# Extract topic-term distributions
def get_prevalent_bigrams(lda_model, dictionary, num_words=8):
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    prevalent_bigrams = {}
    for topic_id, words in topics:
        topic_bigrams = {word: weight for word, weight in words}
        prevalent_bigrams[f"Topic {topic_id + 1}"] = topic_bigrams
    return prevalent_bigrams

# Generate inline HTML word clouds for noun-noun bigrams
def generate_html_wordclouds_bigrams(topic_bigrams, grid_columns=3):
    html_content = "<h1>Word Clouds for LDA Topics (Filtered Noun-Noun Bigrams)</h1>"
    html_content += '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'

    for idx, (topic, bigrams) in enumerate(topic_bigrams.items(), start=1):
        # Generate the word cloud
        wordcloud = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(bigrams)
        
        # Save the word cloud to a BytesIO object
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format="PNG")
        img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        # Add the word cloud to the HTML
        html_content += f"""
        <div style="flex: 1 0 calc(33.33% - 20px); text-align: center; margin-bottom: 20px;">
            <h3>{topic}</h3>
            <img src="data:image/png;base64,{img_data}" alt="{topic}" style="max-width: 100%; height: auto;">
        </div>
        """

    html_content += "</div>"
    return html_content

# Generate prevalent noun-noun bigrams and visualize as inline HTML word clouds
prevalent_bigrams = get_prevalent_bigrams(lda_gensim, dictionary, num_words=15)
wordclouds_html = generate_html_wordclouds_bigrams(prevalent_bigrams, grid_columns=3)

# Combine PyLDAvis HTML and Word Clouds HTML
final_html = f"""
<html>
<head>
    <title>LDA Visualization with Filtered Bigram Word Clouds</title>
</head>
<body>
    {pyLDAvis_html}
    <hr>
    {wordclouds_html}
</body>
</html>
"""

# Save the combined HTML file
output_file = "lda_visualization_with_filtered_bigram_wordclouds.html"
with open(output_file, "w") as f:
    f.write(final_html)

print(f"Final visualization saved as '{output_file}'")
