import pandas as pd
import matplotlib.pyplot as plt
import nltk
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import json
import datetime

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

# Calculate coherence scores for a range of topics, alpha, and eta
def tune_lda(dictionary, corpus, texts, num_topics_range, alpha_vals, eta_vals):
    tuning_results = []
    for alpha in alpha_vals:
        for eta in eta_vals:
            for num_topics in num_topics_range:
                lda_model = gensim.models.LdaModel(
                    corpus=corpus, id2word=dictionary, num_topics=num_topics, 
                    passes=15, alpha=alpha, eta=eta, random_state=42
                )
                coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_score = coherence_model.get_coherence()
                tuning_results.append((num_topics, alpha, eta, coherence_score))
                print(f"Num Topics: {num_topics}, Alpha: {alpha}, Eta: {eta}, Coherence Score: {coherence_score}")
    return tuning_results

# Main function to preprocess, tune hyperparameters, and plot results
def main():
    # Preprocess text for Gensim LDA
    texts = [preprocess_text(description) for description in descriptions]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Define ranges for number of topics, alpha, and eta
    num_topics_range = range(2, 31, 1)
    alpha_vals = ['symmetric', 'asymmetric', 0.01, 0.1, 1]  # 'auto' not available in Gensim for alpha
    eta_vals = ['symmetric', 0.01, 0.1, 1]

    # Run tuning
    tuning_results = tune_lda(dictionary, corpus, texts, num_topics_range, alpha_vals, eta_vals)

    # Save tuning results to JSON
    with open("tuning_results.json", "w") as f:
        json.dump(tuning_results, f)
    print("Tuning results saved to 'tuning_results.json'")

    # Extract results for plotting
    num_topics = [result[0] for result in tuning_results]
    coherence_scores = [result[3] for result in tuning_results]

    # Plot coherence scores
    plt.figure(figsize=(12, 8))
    plt.scatter(num_topics, coherence_scores, marker='o', color='b')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score by Number of Topics (with Alpha and Eta Tuning)")
    plt.grid(True)
    
    # Save the plot as a PNG file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"tuning_coherence_scores_{timestamp}.png"
    plt.savefig(plot_filename, format="png")
    print(f"Plot saved as '{plot_filename}'")

    # Identify best parameters based on coherence score
    best_result = max(tuning_results, key=lambda x: x[3])
    print(f"Best Result - Num Topics: {best_result[0]}, Alpha: {best_result[1]}, Eta: {best_result[2]}, Coherence Score: {best_result[3]}")

# Run the main function inside a main guard
if __name__ == '__main__':
    main()
