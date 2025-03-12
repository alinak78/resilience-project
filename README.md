# Toronto Resilience plan

## Climate Adaptation Solutions - Topic Modeling and N-Grams Analysis

The repository has script (app.py) that analyzes Description column of the 'climate adaptation solutions' csv to identify common topics and key phrases. Using Python libraries like **NLTK**, **Sklearn**, **Gensim**, and **PyLDAvis**, it performs text preprocessing, generates bigrams, and applies topic modeling with **Latent Dirichlet Allocation (LDA)**. The results are visualized interactively, allowing users to explore topics and related terms.

Demo:
![alt text]([http://url/to/img.png](https://github.com/alinak78/resilience-project/blob/main/topic_model.gif))

### Key Features
- **Data Loading and Preprocessing**: Cleans and tokenizes text descriptions.
- **Bigram Generation**: Extracts frequent two-word phrases for insights into common themes.
- **Topic Modeling with LDA**: Generates topics using Sklearn and Gensim LDA models.
- **Interactive Visualization**: Saves an HTML visualization for exploring topics and keywords.
- **Documentation**: Comprehensive documentation with code explanations for easy reference.

### Visualization map:
**1. Intertopic Distance Map (Left Panel)**
Purpose: This map provides a visual representation of the topics and their relationships to each other, using Multidimensional Scaling (MDS) to plot them in 2D space.
Circles: Each circle represents a topic. The size of the circle indicates the prevalence or relative frequency of the topic in the corpus.
Distances:
The distance between circles shows how closely related the topics are. Closer circles indicate that topics share more words in common, while distant circles are more distinct.
Selected Topic: In this screenshot, Topic 3 is selected and highlighted in red. This selection affects the terms displayed in the right panel.
[pic 1](images/pic1.png)

**2. Top-30 Most Relevant Terms for Selected Topic (Right Panel)**
Purpose: This bar chart shows the top terms most relevant to the currently selected topic, in this case, Topic 3.
Bars:
Red bars represent the frequency of each term within the selected topic.
Blue bars represent the overall frequency of each term across the entire corpus.
The length of the red bar relative to the blue bar indicates how much the term is specific to this topic compared to the whole corpus.
Terms: The terms help interpret the topic's content. For example, terms like "waste," "management," "system," "sanitation," and "water" suggest that Topic 3 might relate to waste management and sanitation.

**3. Lambda (λ) Slider (Top Right)**
Purpose: The λ slider controls the relevance metric for displaying terms in the selected topic.
Relevance Metric (λ):
λ is a parameter that adjusts the balance between term frequency (how often a term appears) and exclusivity(how unique a term is to the topic).
A λ value of 1 shows terms that are highly frequent within the topic (possibly shared with other topics).
A λ value of 0 shows terms that are highly exclusive to the topic (even if infrequent).
In the example (λ = 0.6), the relevance metric favors terms that are relatively frequent in the topic but still somewhat exclusive.

**4. Marginal Topic Distribution (Bottom Left)**
Purpose: This small inset visualizes the overall distribution of topics in the corpus, with circle sizes corresponding to topic prevalence.
Interpretation: It helps quickly understand which topics are most common in the corpus. Larger circles indicate topics that occupy a higher percentage of the overall text.
Key Terms in the Explanation (Bottom Right)
Saliency: Indicates the importance of a term in the overall corpus, balancing frequency and relevance.
Relevance: Calculated as the relevance of a term within a topic, based on λ and the term’s exclusivity and frequency.
Summary Interpretation of This Visualization
In this visualization:
Topic 3 (highlighted in red) appears as a distinct, moderately prevalent topic.
Based on the right panel's terms, Topic 3 is likely about waste management, sanitation, and related infrastructure.
Adjusting λ lets you explore terms that are more frequent or exclusive, helping you gain nuanced insights into what defines each topic.






