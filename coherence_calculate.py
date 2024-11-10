import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import json

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load the CSV file
df = pd.read_csv("climate_adaptation_solutions_full.csv")
descriptions = df['Description'].dropna().tolist()

# Preprocess text function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Calculate coherence scores for a range of topics
from gensim.models.coherencemodel import CoherenceModel
import gensim

def tune_lda(dictionary, corpus, texts, start=2, end=30, step=1, alpha='symmetric', eta='symmetric'):
    coherence_scores = []
    for num_topics in range(start, end + 1, step):
        lda_model = gensim.models.LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15,
            alpha=alpha, eta=eta, random_state=42
        )
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append((num_topics, coherence_score))
        print(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}")
    return coherence_scores

# Example of tuning alpha and eta


def calculate_coherence(dictionary, corpus, texts, start=2, end=30, step=1):
    coherence_scores = []
    for num_topics in range(start, end + 1, step):
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append((num_topics, coherence_score))
        print(f"Number of Topics: {num_topics}, Coherence Score: {coherence_score}")
    return coherence_scores

# Main function to preprocess and calculate coherence
def main():
    # Preprocess text for Gensim LDA
    texts = [preprocess_text(description) for description in descriptions]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Calculate coherence scores
    coherence_scores = calculate_coherence(dictionary, corpus, texts, start=2, end=30, step=1)
    #coherence_scores = tune_lda(dictionary, corpus, texts, start=5, end=20, alpha='auto', eta='auto')

    # Extract topic numbers and coherence scores for plotting
    topic_nums = [score[0] for score in coherence_scores]
    coherence_vals = [score[1] for score in coherence_scores]

    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(topic_nums, coherence_vals, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score by Number of Topics")
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig("coherence_scores.png", format="png")
    print("Plot saved as 'coherence_scores.png'")

# Run the main function inside a main guard
if __name__ == '__main__':
    main()
