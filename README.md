# Information Retrieval System for Video Game Dataset

Thisis a Python-based Information Retrieval (IR) System it is made to be able to search
and rank video games from a dataset of 399 HTML pages. The system supports various query
handling techniques and preprocessing options and evaluates performance using metrics like 
Precision@10 and Average Similarity.

---

## Features

1. Preprocessing Techniques:
   - Tokenization, Stopword Removal, Stemming, and Lemmatization.
   - Metadata extraction (e.g., genres, publishers, developers, age ratings).

2. Query Handling:
   - Basic term-based querying.
   - Metadata-driven query expansion.
   - Named Entity Recognition for publishers and developers.

3. Search Engine Functionalities:
   - Computes TF-IDF for documents.
   - Ranks documents using Cosine Similarity.
   - Metadata-weighted ranking to boost relevant matches.

4. Evaluation Metrics:
   - Precision@10: Percentage of relevant results in the top 10.
   - Average Similarity: Measures similarity across all results.


5. CLI Interface:
   - Change preprocessing settings.
   - Perform search queries.
   - Evaluate predefined queries with Precision@10.

---
1. SETUP INSTRUCTIONS
   - pip install -r requirements.txt
   - Download NLTK and SpaCy resources:
   - import nltk
   - nltk.download('wordnet')
   - nltk.download('omw-1.4')
   - python -m spacy download en_core_web_sm

USAGE
   - Run the code - python baseline.py

CLI Interface controls,
1. Preprocessing menu
    - 1 to change preprocessign settings. #NOTE delete -preprocessed_data.pkl if you are changing the settings 
    - 2 to build preprocessed_data.pkl
    - 3 to exit
2. Query Searching menu
    - 1 to search a query
    - 2 to evaluate system with 10 preset queires
    - exit to quit

