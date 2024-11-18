import pandas as pd
import nltk
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud
from io import BytesIO
import base64
from collections import Counter

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.util import ngrams

# Load the CSV file
df = pd.read_csv("climate_adaptation_solutions-2.csv")

# Use only the 'Description' column
if 'Description' in df.columns:
    descriptions = df['Description'].dropna().tolist()
else:
    raise ValueError("'Description' column is missing in the dataset.")

# Custom stopwords
#custom_stopwords = set(stopwords.words('english')).union({"climate", "solution", "adaptation", "solutions", "project"})

# Preprocess text function (noun-noun bigrams only)
def preprocess_text_with_noun_noun_bigrams(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    #tokens = [word for word in tokens if word not in custom_stopwords]
    
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Extract noun-noun bigrams
    bigrams = [
        f"{word1} {word2}"
        for (word1, tag1), (word2, tag2) in ngrams(pos_tags, 2)
        if tag1.startswith('NN') and tag2.startswith('NN')
    ]
    return bigrams

# Preprocess all descriptions to noun-noun bigrams
texts = [preprocess_text_with_noun_noun_bigrams(text) for text in descriptions]

# Flatten the bigrams for frequency analysis
all_bigrams = [bigram for text in texts for bigram in text]

# Analyze most common bigrams for additional stopwords
bigram_counts = Counter(all_bigrams)
print("Most common noun-noun bigrams:", bigram_counts.most_common(20))

# Gensim LDA for topic modeling (with noun-noun bigrams)
def gensim_lda(texts, num_topics=10, passes=15):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)
    return lda_model, corpus, dictionary

lda_gensim, corpus, dictionary = gensim_lda(texts)

# Save Gensim LDA visualization as HTML
lda_display = pyLDAvis.gensim.prepare(lda_gensim, corpus, dictionary)
pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_display)

# Extract topic-term distributions with saliency filtering
def get_prevalent_noun_noun_bigrams_with_saliency(lda_model, dictionary, num_words=8, saliency_threshold=0.2):
    topics = lda_model.get_topics()  # Topic-word probabilities
    num_topics = topics.shape[0]
    bigram_saliency = {}

    # Calculate saliency for each bigram
    for word_id in range(topics.shape[1]):
        total_weight = topics[:, word_id].sum()
        for topic_idx in range(num_topics):
            word = dictionary[word_id]
            saliency = topics[topic_idx, word_id] / total_weight
            bigram_saliency[word] = max(bigram_saliency.get(word, 0), saliency)

    # Filter bigrams by saliency
    prevalent_bigrams = {}
    for topic_idx in range(num_topics):
        topic_bigrams = {
            dictionary[word_id]: topics[topic_idx, word_id]
            for word_id in range(topics.shape[1])
            if bigram_saliency[dictionary[word_id]] >= saliency_threshold
        }
        # Keep only the top `num_words` bigrams for each topic
        prevalent_bigrams[f"Topic {topic_idx + 1}"] = dict(
            sorted(topic_bigrams.items(), key=lambda x: x[1], reverse=True)[:num_words]
        )

    return prevalent_bigrams

# Generate inline HTML word clouds for noun-noun bigrams
def generate_html_wordclouds_bigrams(topic_bigrams, grid_columns=3):
    html_content = "<h1>Word Clouds for LDA Topics (Filtered Noun-Noun Bigrams by Saliency)</h1>"
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

# Generate prevalent noun-noun bigrams filtered by saliency and visualize as inline HTML word clouds
prevalent_bigrams = get_prevalent_noun_noun_bigrams_with_saliency(lda_gensim, dictionary, num_words=15, saliency_threshold=0.3)
wordclouds_html = generate_html_wordclouds_bigrams(prevalent_bigrams, grid_columns=3)

# Combine PyLDAvis HTML and Word Clouds HTML
final_html = f"""
<html>
<head>
    <title>LDA Visualization with Noun-Noun Bigram Word Clouds</title>
</head>
<body>
    {pyLDAvis_html}
    <hr>
    {wordclouds_html}
</body>
</html>
"""

# Save the combined HTML file
output_file = "new/lda_visualization_with_filtered_bigram_wordclouds.html"
with open(output_file, "w") as f:
    f.write(final_html)

print(f"Final visualization saved as '{output_file}'")
