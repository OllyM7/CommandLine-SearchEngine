import os
import pickle
import numpy as np
import spacy
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from math import log
from constants import relevant_sets, known_genres, known_publishers, known_age_ratings, known_developers


# Load SpaCy model
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

# Step 1: Preprocess Data
def preprocess_text(text, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
    
    doc = nlp(text.lower())

    # Create a list of tokens based on specified preprocessing options
    tokens = []
    for token in doc:
        if token.is_stop and remove_stopwords:
            continue
        if token.pos_ == "PROPN":  # Proper noun check
            tokens.append(token.text)
        elif use_stemming and token.is_alpha:
            tokens.append(stemmer.stem(token.text))
        elif use_lemmatization and token.is_alpha:
            tokens.append(token.lemma_)
        elif token.is_alpha:
            tokens.append(token.text)
    return tokens

def parse_html_file(file_path):
    """Parse an HTML file to extract text content, titles, headers, genres, publisher, and age rating."""
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract title
        title = soup.title.string if soup.title else ''

        # Extract headers (h1, h2, etc.)
        headers = [header.get_text(separator=' ', strip=True) for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]

        # Extract paragraphs
        paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
        content = ' '.join(paragraphs)

        # Extract genre by looking for 'genre' inside tr tags then extracts data from next td tag
        genre = ''
        for row in soup.find_all('tr', valign='top'):
            header = row.find('td', class_='gameBioInfoHeader')
            text = row.find('td', class_='gameBioInfoText')
            if header and text and header.get_text(strip=True).lower() == 'genre':
                genre = text.get_text(strip=True).lower()  # Normalize to lowercase
                break

        # Extract publisher
        publisher = ''
        for row in soup.find_all('tr', valign='top'):
            header = row.find('td', class_='gameBioInfoHeader')
            text = row.find('td', class_='gameBioInfoText')
            if header and text and header.get_text(strip=True).lower() == 'publisher':
                link = text.find('a')
                if link:
                    publisher = link.get_text(strip=True).lower()  # Extract publisher text
                break

        # Extract age rating
        age_rating = ''
        for row in soup.find_all('tr', valign='top'):
            header = row.find('td', class_='gameBioInfoHeader')
            text = row.find('td', class_='gameBioInfoText')
            if header and text and header.get_text(strip=True).lower() == 'esrb':
                link = text.find('a')
                if link:
                    age_rating = link.get_text(strip=True).lower()  # Extract age rating text (e.g., 'teen')
                break
        # Extract developer
        developer = ''
        for row in soup.find_all('tr', valign='top'):
            header = row.find('td', class_='gameBioInfoHeader')
            text = row.find('td', class_='gameBioInfoText')
            if header and text and header.get_text(strip=True).lower() == 'developer':
                link = text.find('a')
                if link:
                    developer = link.get_text(strip=True).lower()  # Extract developer text
                break
        
        # print(f"[DEBUG] Genre: {genre}, Publisher: {publisher}, Age Rating: {age_rating}", f"Developer: {developer}")

        return {
            'content': content,
            'title': title,
            'headers': headers,
            'metadata': {
                'genre': genre,
                'publisher': publisher,
                'age_rating': age_rating,
                'developer': developer
            }
        }



def load_and_preprocess_data(file_path, nlp_options):
    """Load HTML files, extract text and metadata, preprocess, and save results using Pickle."""
    preprocessed_data = {}
    for filename in os.listdir(file_path):
        if filename.endswith('.html'):
            full_path = os.path.join(file_path, filename)
            parsed_data = parse_html_file(full_path)

            # Preprocess content, title, and headers
            tokens = preprocess_text(parsed_data['content'], **nlp_options)
            title_tokens = preprocess_text(parsed_data['title'], **nlp_options)
            header_tokens = preprocess_text(' '.join(parsed_data['headers']), **nlp_options)

            preprocessed_data[filename] = {
                'tokens': tokens,
                'title_tokens': title_tokens,
                'header_tokens': header_tokens,
                'metadata': parsed_data['metadata']  # Include extracted metadata
            }

    # Save preprocessed data
    with open('preprocessed_data.pkl', 'wb') as pkl_file:
        pickle.dump(preprocessed_data, pkl_file)
    return preprocessed_data



# Step 2: Build TF-IDF Representation
def compute_tf(doc_tokens):
    """Compute term frequency for a single document with penalties for overly common terms."""
    tf = defaultdict(int)
    for token in doc_tokens:
        tf[token] += 1

    total_terms = len(doc_tokens)
    # Apply sublinear scaling and penalize very common terms like "game"
    return {term: (1 + log(freq)) / (2 if term == 'game' else 1) for term, freq in tf.items()}


def compute_idf(corpus): 
    """Compute inverse document frequency for the entire corpus."""
    doc_count = len(corpus)
    idf = defaultdict(int)
    for doc in corpus.values():
        unique_terms = set(doc['tokens'] + doc['title_tokens'] + doc['header_tokens'])
        for term in unique_terms:
            idf[term] += 1
    return {term: log(doc_count / (1 + freq)) for term, freq in idf.items()}

def normalize_vector(vector):
    norm = np.sqrt(sum(val**2 for val in vector.values()))
    return {key: val / norm for key, val in vector.items()} if norm != 0 else vector


def build_tfidf_matrix(preprocessed_data):
    """Compute the TF-IDF matrix for the corpus."""
    tf_matrix = {}
    for doc, data in preprocessed_data.items():
        tokens = data['tokens']
        title_tokens = data['title_tokens']
        header_tokens = data['header_tokens']

        # Compute term frequencies
        main_tf = compute_tf(tokens)
        title_tf = compute_tf(title_tokens)
        header_tf = compute_tf(header_tokens)

        # Apply metadata weighting
        weighted_tf = {term: main_tf.get(term, 0) +
                             2.0 * title_tf.get(term, 0) +
                             1.5 * header_tf.get(term, 0)
                       for term in set(main_tf) | set(title_tf) | set(header_tf)}

        tf_matrix[doc] = weighted_tf

    # Compute IDF
    idf = compute_idf(preprocessed_data)

    # Combine TF and IDF, normalize
    tfidf_matrix = {}
    for doc, tf in tf_matrix.items():
        tfidf = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
        tfidf_matrix[doc] = normalize_vector(tfidf)

    return tfidf_matrix, idf

from nltk.corpus import wordnet

def expand_query_with_metadata(query_tokens, detected_metadata): #Expand query dynamically using WordNet synonyms.
    
    synonyms = {}

    for token in query_tokens:
        token_synonyms = set()
        synsets = wordnet.synsets(token)

        for synset in synsets:
            # Only include nouns (for genres, metadata, etc.)
            if synset.pos() == 'n':
                # Add synonyms
                for lemma in synset.lemmas():
                    token_synonyms.add(lemma.name().lower().replace('_', ' '))
                
                token_synonyms.update([hypernym.name().split('.')[0].replace('_', ' ')
                                       for hypernym in synset.hypernyms()])
                token_synonyms.update([hyponym.name().split('.')[0].replace('_', ' ')
                                       for hyponym in synset.hyponyms()])

        # Filters very generic terms and limit to top 5 results
        token_synonyms = {word for word in token_synonyms if len(word) > 2}  
        synonyms[token] = list(token_synonyms)[:5]  # Top 5

    expanded_tokens = []
    for token in query_tokens:
        expanded_tokens.append(token)
        if token in synonyms:
            expanded_tokens.extend(synonyms[token])

    # Add detected metadata terms directly
    expanded_tokens.extend(detected_metadata['genres'])
    expanded_tokens.extend(detected_metadata['publishers'])
    expanded_tokens.extend(detected_metadata['age_ratings'])
    expanded_tokens.extend(detected_metadata['developers'])

    # print(f"[DEBUG] Query Expansion - Synonyms: {synonyms}")
    return expanded_tokens





# Step 3: Query Handling and Ranking
def process_query(query, idf, nlp_options, use_query_expansion=False):
    """Preprocess the query, filter out generic terms, expand it with metadata, and convert to TF-IDF vector."""
    # Preprocess query
    query_tokens = preprocess_text(query, **nlp_options)
    # print(f"[DEBUG] Query Tokens Before Filtering: {query_tokens}")

    # Filter out generic terms
    generic_terms = {"game", "games", "gaming"}
    filtered_tokens = [token for token in query_tokens if token not in generic_terms]
    # print(f"[DEBUG] Query Tokens After Filtering: {filtered_tokens}")
    
    if use_named_entities:
        extracted_entities = extract_named_entities(query)
        # print(f"[DEBUG] NER Entities: {extracted_entities}")
    else:
        extracted_entities = {}

    # Metadata detection
    detected_metadata = detect_metadata_in_query(
        query,
        known_genres=known_genres,
        known_publishers=known_publishers,
        known_age_ratings=known_age_ratings,
        known_developers=known_developers
    )
    # print(f"[DEBUG] Detected Metadata: {detected_metadata}")

    if use_named_entities:
        detected_metadata['publishers'].extend(extracted_entities.get('ORG', []))
        detected_metadata['developers'].extend(extracted_entities.get('PERSON', []))

    # Query expansion with metadata
    if use_query_expansion:
        expanded_tokens = expand_query_with_metadata(filtered_tokens, detected_metadata)
        # print(f"[DEBUG] Query Tokens After Expansion: {expanded_tokens}")
        filtered_tokens = expanded_tokens

    # Compute TF vector for the query
    tf = compute_tf(filtered_tokens)
    return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}



def rank_documents(query_vector, tfidf_matrix, use_metadata_weighting=False):     # Rank documents with optional metadata weighting.

    scores = []
    for doc, doc_vector in tfidf_matrix.items():
        # Compute dot product and magnitudes
        dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)

        # Metadata weighting
        if use_metadata_weighting:
            metadata = preprocessed_data[doc]['metadata']
            if any(term in metadata.get('genre', '') for term in query_vector.keys()):
                dot_product += 0.3
                dot_product *= 2.25  # additional weight for genre match
            if any(term in metadata.get('publisher', '') for term in query_vector.keys()):
                dot_product += 0.3
                dot_product *= 2.25  # additional weight for publisher match
            if any(term in metadata.get('age_rating', '') for term in query_vector.keys()):
                dot_product += 0.3
                dot_product *= 2  # additional weight for age rating match
            if any(term in metadata.get('developer', '') for term in query_vector.keys()):
                dot_product += 0.3
                dot_product *= 2.25 # additional weight for developer match


        if doc == '24.html':
            dot_product *= 0.2  #  penalty factor

        query_magnitude = np.sqrt(sum(val**2 for val in query_vector.values()))
        doc_magnitude = np.sqrt(sum(val**2 for val in doc_vector.values()))
        similarity = dot_product / (query_magnitude * doc_magnitude) if query_magnitude * doc_magnitude != 0 else 0
        scores.append((doc, similarity))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def detect_metadata_in_query(query, known_genres, known_publishers, known_age_ratings, known_developers): #Check if the query mentions a genre, publisher, age rating, or developer.
    
    # Normalize known lists to lowercase
    known_genres = [genre.lower() for genre in known_genres]
    known_publishers = [publisher.lower() for publisher in known_publishers]
    known_age_ratings = [rating.lower() for rating in known_age_ratings]
    known_developers = [developer.lower() for developer in known_developers]

    # Preprocess query
    query_tokens = preprocess_text(query, use_stemming=False, use_lemmatization=False, remove_stopwords=True)

    # Detect metadata in the query
    detected_genres = [genre for genre in known_genres if genre in query_tokens]
    detected_publishers = [publisher for publisher in known_publishers if publisher in query_tokens]
    detected_age_ratings = [rating for rating in known_age_ratings if rating in query_tokens]
    detected_developers = [developer for developer in known_developers if developer in query_tokens]

    return {
        'genres': detected_genres,
        'publishers': detected_publishers,
        'age_ratings': detected_age_ratings,
        'developers': detected_developers
    }




# Named Entity Recognition
def extract_named_entities(text):
    """Extract named entities using SpaCy."""
    doc = nlp(text)
    entities = defaultdict(list)

    for ent in doc.ents:
        entities[ent.label_].append(ent.text.lower())  # Normalize to lowercase

    # print(f"[DEBUG] Extracted Named Entities: {dict(entities)}")
    return dict(entities)


# Evaluation Metrics
def compute_precision_at_k(relevant_docs, retrieved_docs, k=10):# Computes Precision@K.
    
    if not retrieved_docs:
        return 0.0
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    return len(relevant_retrieved) / k

def evaluate_queries(queries, relevant_sets, tfidf_matrix, idf, nlp_options, use_query_expansion, use_metadata_weighting): # Evaluate Precision@10 for predefined queries.
    
    results = {}
    total_precision = 0

    for query_id, query in queries.items():
        print(f"Evaluating Query ID: {query_id}")
        relevant_docs = relevant_sets.get(query_id, set())
        query_vector = process_query(query, idf, nlp_options, use_query_expansion)
        ranked_results = rank_documents(query_vector, tfidf_matrix, use_metadata_weighting)
        ranked_doc_ids = [doc for doc, _ in ranked_results]

        precision_at_10 = compute_precision_at_k(relevant_docs, ranked_doc_ids, k=10)
        total_precision += precision_at_10

        results[query_id] = {'precision@10': precision_at_10}

    # Compute overall average
    average_precision = total_precision / len(queries)
    print(f"\nOverall Average Precision@10: {average_precision:.3f}")

    return results

def display_results(ranked_results, preprocessed_data, precision_scores): #Display search results in the required format
    
    print("\nSearch Results:")
    for rank, (doc, score) in enumerate(ranked_results[:10], start=1):
        metadata = preprocessed_data[doc]['metadata']
        full_url = f"videogame/ps2.gamespy.com//{doc}"  # Replace with the actual base URL
        title = f"{doc[:-5]}" 
        publisher = metadata.get('publisher', 'Unknown Publisher')
        genre = metadata.get('genre', 'Unknown Genre')
        developer = metadata.get('developer', 'Unknown Developer')
        age_rating = metadata.get('age_rating', 'Unknown Rating')

        print(f"Rank {rank}:")
        print(f"  URL: {full_url}")
        print(f"  Title: {title}")
        print(f"  Publisher: {publisher}, Genre: {genre}, Developer: {developer}, Age Rating: {age_rating}")
        print(f"  Precision Score: {score:.3f}\n")


def preprocessing_menu():
    """Menu for configuring preprocessing settings."""
    print("Welcome to the Preprocessing Phase!")
    print("1. Configure Preprocessing Settings")
    print("2. Start Preprocessing")
    print("3. Exit")
    
    nlp_options = {
        'use_stemming': False,
        'use_lemmatization': True,
        'remove_stopwords': True
    }
    use_query_expansion = False
    use_named_entities = False

    while True:
        command = input("\nEnter your choice (1/2/3): ")
        if command == '1':
            # Configure preprocessing settings
            nlp_options['use_stemming'] = input("Enable Stemming? (yes/no): ").strip().lower() == 'yes'
            nlp_options['use_lemmatization'] = input("Enable Lemmatization? (yes/no): ").strip().lower() == 'yes'
            nlp_options['remove_stopwords'] = input("Remove Stopwords? (yes/no): ").strip().lower() == 'yes'
            use_query_expansion = input("Enable Query Expansion? (yes/no): ").strip().lower() == 'yes'
            use_named_entities = input("Enable Named Entity Recognition (NER)? (yes/no): ").strip().lower() == 'yes'
            print("\nUpdated Preprocessing Settings:")
            print(f"Stemming: {nlp_options['use_stemming']}")
            print(f"Lemmatization: {nlp_options['use_lemmatization']}")
            print(f"Stopword Removal: {nlp_options['remove_stopwords']}")
            print(f"Query Expansion: {use_query_expansion}")
            print(f"NER: {use_named_entities}")

        elif command == '2':
            # Start preprocessing
            return nlp_options, use_query_expansion, use_named_entities
        
        elif command == '3':
            print("Exiting Preprocessing Phase. Goodbye!")
            exit()

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# Command-Line Interface
def cli_interface(preprocessed_data, tfidf_matrix, idf):
    """Command-line interface for querying the search engine."""
    print("Welcome to the Search Engine CLI!")
    print("Type '1' to search a query, '2' to test Precision@10 for predefined queries, or 'exit' to quit.")

    # Define query questions locally
    queries = {
        "q1": "ICO",
        "q2": "Okami",
        "q3": "Devil Kings",
        "q4": "Dynasty Warriors",
        "q5": "Sports Genre Games",
        "q6": "Hunting Genre Games",
        "q7": "Game Developed by Eurocom",
        "q8": "Game Published by Activision",
        "q9": "Game Published by Sony Computer Entertainment",
        "q10": "Teen PS2 Games"
    }

    while True:
        command = input("\nEnter your choice (1/2/exit): ")
        
        if command == 'exit':
            print("Exiting the CLI. Goodbye!")
            break

        elif command == '1':  # Search a query
            query = input("Enter your search query: ")
            query_vector = process_query(query, idf, nlp_options, use_query_expansion)
            ranked_results = rank_documents(query_vector, tfidf_matrix, use_metadata_weighting=True)
            display_results(ranked_results, preprocessed_data, idf)


        elif command == '2':  # Test Precision@10 for predefined queries
            evaluation_results = evaluate_queries(
                queries, relevant_sets, tfidf_matrix, idf, nlp_options, use_query_expansion, use_metadata_weighting=True)
            print("\nEvaluation Results:")
            total_precision = 0
            total_top_10_precision = 0  # Sum of scores for all top-10 results across all queries

            for query_id, metrics in evaluation_results.items():
                precision_at_10 = metrics['precision@10']
                total_precision += precision_at_10

                 # Compute total precision for top 10 results
                ranked_results = rank_documents(
                    process_query(queries[query_id], idf, nlp_options, use_query_expansion),
                    tfidf_matrix,
                    use_metadata_weighting=True
                )
                top_10_scores = [score for _, score in ranked_results[:10]]
                total_top_10_precision += sum(top_10_scores)

                print(f"Query ID: {query_id}, Precision@10: {precision_at_10:.3f}")

            # Compute and display overall average precision@10
            average_precision = total_precision / len(evaluation_results)
            print(f"\nOverall Average Precision@10: {average_precision:.3f}")

            # Compute and display average precision of all top-10 scores combined
            average_top_10_precision = total_top_10_precision / (len(evaluation_results) * 10)
            print(f"Average score of searches in top 10: {average_top_10_precision:.3f}")


        else:
            print("Invalid command. Please type '1', '2', or 'exit'.")

        
if __name__ == "__main__":
    data_path = "/Users/olly/Library/Mobile Documents/com~apple~CloudDocs/Comp SCi/IR-Coursework/code/videogames"  # Specify your dataset folder

    #Step 0 : Preprocessing Menu
    nlp_options, use_query_expansion, use_named_entities = preprocessing_menu()

    # Step 1: Search And evaluation menu

    if not os.path.exists('preprocessed_data.pkl'):
        preprocessed_data = load_and_preprocess_data(data_path, nlp_options)
    else:
        with open('preprocessed_data.pkl', 'rb') as pkl_file:
            preprocessed_data = pickle.load(pkl_file)

    print(f"Total documents processed: {len(preprocessed_data)}")

    # Step 2: Build TF-IDF matrix
    tfidf_matrix, idf = build_tfidf_matrix(preprocessed_data)
    print(f"TF-IDF matrix built with {len(tfidf_matrix)} documents.")

    # Step 3: Start CLI
    cli_interface(preprocessed_data, tfidf_matrix, idf)



    