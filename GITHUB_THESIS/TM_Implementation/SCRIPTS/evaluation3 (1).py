import json
import time
import itertools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from typing import Mapping, Any, List, Tuple

try:
    from bertopic import BERTopic
except ImportError:
    pass

try:
    from top2vec import Top2Vec
except ImportError:
    pass

try:
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import (
        TopicModelDataPreparation,
    )
    import nltk

    nltk.download("stopwords")
    from nltk.corpus import stopwords
except ImportError:
    pass

from octis.models.ETM import ETM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.models.CTM import CTM
from octis.models.NeuralLDA import NeuralLDA
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

import gensim
import gensim.corpora as corpora
from gensim.models import ldaseqmodel


def read_stop_words(file_path):
    """Read stop words from a file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file]
    return stop_words

class CustomOctisDataset:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_corpus(self):
        return [doc.split() for doc in self.corpus]

class Trainer:
    """Train and evaluate a topic model"""

    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: Mapping[str, Any],
        topk: int = 10,
        custom_dataset: bool = False,
        bt_embeddings: np.ndarray = None,
        bt_timestamps: List[str] = None,
        bt_nr_bins: int = None,
        custom_model=None,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name
        self.params = params
        self.topk = topk
        self.timestamps = bt_timestamps
        self.nr_bins = bt_nr_bins
        self.embeddings = bt_embeddings
        self.ctm_preprocessed_docs = None
        self.custom_model = custom_model
        self.verbose = verbose

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    def train(self, save: str = False) -> Mapping[str, Any]:
        """Train a topic model"""

        results = []

        # Loop over all parameters
        params_name = list(self.params.keys())
        params = {
            param: (value if type(value) == list else [value])
            for param, value in self.params.items()
        }
        new_params = list(itertools.product(*params.values()))
        for param_combo in new_params:

            # Train and evaluate model
            params_to_use = {
                param: value for param, value in zip(params_name, param_combo)
            }
            output, computation_time, topics = self._train_tm_model(params_to_use)
            scores = self.evaluate(output)

            # Update results
            result = {
                "Dataset": self.dataset,
                "Dataset Size": len(self.data.get_corpus()),
                "Model": self.model_name,
                "Scores": scores,
                "Computation Time": computation_time,
                "Topics": topics,
            }
            results.append(result)

        if save:
            with open(f"{save}.json", "w") as f:
                json.dump(results, f)

            try:
                from google.colab import files
                files.download(f"{save}.json")
            except ImportError:
                pass

        return results

    def _train_tm_model(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float, List[List[str]]]:
        """Select and train the Topic Model"""
        if self.model_name == "BERTopic":
            output, computation_time, topics = self._train_bertopic(params)
            return output, computation_time, topics
        
        elif self.model_name == "CTM_CUSTOM":
            if self.qt_ctm is None:
                self._preprocess_ctm()
            output, computation_time, topics = self._train_ctm(params)
            return output, computation_time, topics
         

        # Train OCTIS model
        # Train OCTIS model
        octis_models = ["ETM", "LDA", "CTM", "NMF", "NeuralLDA"]
        if self.model_name in octis_models:
            output, computation_time, topics = self._train_octis_model(params)
            return output, computation_time, topics     

    def _train_ctm(self, params) -> Tuple[Mapping[str, Any], float]:
        """Train CTM"""
        params["bow_size"] = len(self.qt_ctm.vocab)
        ctm = CombinedTM(**params)

        start = time.time()
        ctm.fit(self.training_dataset_ctm)
        end = time.time()
        computation_time = float(end - start)

        topics = ctm.get_topics(10)
        topics = [topics[x] for x in topics]

        output_tm = {
            "topics": topics,
        }

        return output_tm, computation_time, topics

    def _preprocess_ctm(self):
        
        """Preprocess data for CTM"""
        import re
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import nltk
        # Function to read stopwords from a file
        def read_stop_words(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                custom_stopwords = file.read().splitlines()
                return custom_stopwords

        # Path to your custom stopwords file
        stop_words_file = "stopwords-nl.txt"

        # Read custom stopwords
        historical_stopwords = read_stop_words(stop_words_file)

        # Initialize CountVectorizer with custom stopwords
        dutch_stopwords = stopwords.words('dutch')
        english_stopwords = stopwords.words('english')
        # Combine the custom stopwords with the NLTK Dutch and English stopwords
        combined_stopwords = set(dutch_stopwords).union(set(english_stopwords)).union(set(historical_stopwords))
        
        #stop_words = stopwords.words('english')
        
        # Prepare docs
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]
        
        preprocessed_documents = [
            " ".join([x for x in doc.split(" ") if x not in combined_stopwords]).strip()
            for doc in docs
        ]
        

        # Get vocabulary
        vectorizer = CountVectorizer(
            max_features=2000, token_pattern=r"\b[a-zA-Z]{2,}\b"
        )
        vectorizer.fit_transform(preprocessed_documents)
        vocabulary = set(vectorizer.get_feature_names())

        # Preprocess documents further
        preprocessed_documents = [
            " ".join([w for w in doc.split() if w in vocabulary]).strip()
            for doc in preprocessed_documents
        ]

        # Prepare CTM data
        qt = TopicModelDataPreparation("all-mpnet-base-v2")  #jegormeister/bert-base-dutch-cased-snli
        training_dataset = qt.fit(
            text_for_contextual=docs, text_for_bow=preprocessed_documents
        )

        self.qt_ctm = qt
        self.training_dataset_ctm = training_dataset

    def _get_topics_words(self, model, topk):
        """
        Return the most significant words for each topic.
        """
        topic_terms = []
        for i in range(model.hyperparameters["num_topics"]):
            topic_words_list = []
            for word_tuple in model.trained_model.get_topic_terms(i, topk):
                topic_words_list.append(model.id2word[word_tuple[0]])
            topic_terms.append(topic_words_list)
        return topic_terms
    
    def _preprocess_octis(self):
        """Preprocess data for octis topic modeling"""

        import re
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import nltk
        # Function to read stopwords from a file
        def read_stop_words(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                custom_stopwords = file.read().splitlines()
                return custom_stopwords

        # Path to your custom stopwords file
        stop_words_file = "stopwords-nl.txt"

        # Read custom stopwords
        historical_stopwords = read_stop_words(stop_words_file)

        # Initialize CountVectorizer with custom stopwords
        dutch_stopwords = stopwords.words('dutch')
        english_stopwords = stopwords.words('english')
        # Combine the custom stopwords with the NLTK Dutch and English stopwords
        combined_stopwords = set(dutch_stopwords).union(set(english_stopwords)).union(set(historical_stopwords))

        # Prepare docs
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        # Preprocess documents
        preprocessed_documents = []
        for doc in docs:
            # Remove numbers, abbreviations, and unknown characters
            doc = re.sub(r'\b\w{1,2}\b', '', doc)  # Remove abbreviations
            doc = re.sub(r'\d+', '', doc)  # Remove numbers
            doc = re.sub(r'[^a-zA-Z\s]', '', doc)  # Remove special characters

            # Tokenize
            words = word_tokenize(doc.lower())

            # Remove stopwords and join back into string
            words = [word for word in words if word not in combined_stopwords]
            preprocessed_doc = " ".join(words)
            preprocessed_documents.append(preprocessed_doc.strip())

        # Get vocabulary
        vectorizer = CountVectorizer(max_features=2000, token_pattern=r"\b[a-zA-Z]{2,}\b")
        vectorizer.fit_transform(preprocessed_documents)
        vocabulary = set(vectorizer.get_feature_names_out())

        # Further filter documents to keep only words in vocabulary
        final_preprocessed_documents = [
            " ".join([word for word in doc.split() if word in vocabulary]).strip()
            for doc in preprocessed_documents
        ]

        return CustomOctisDataset(final_preprocessed_documents)
    
    def _train_octis_model(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train OCTIS model"""
        # Preprocess data
        preprocessed_data = self._preprocess_octis()

        if self.model_name == "ETM":
            model = ETM(**params)
            model.use_partitions = False
        elif self.model_name == "LDA":
            model = LDA(**params)
            model.use_partitions = False
        elif self.model_name == "CTM":
            model = CTM(**params)
            model.use_partitions = False
        elif self.model_name == "NMF":
            model = NMF(**params)
            model.use_partitions = False
        elif self.model_name == "NeuralLDA":
            model = NeuralLDA(**params)
            model.use_partitions = False

        start = time.time()
        output_tm = model.train_model(preprocessed_data)
        end = time.time()
        computation_time = end - start
        
        topics = self._get_topics_words(model, 10)
        return output_tm, computation_time, topics

    def _train_bertopic(self, params: Mapping[str, Any]) -> Tuple[Mapping[str, Any], float, List[List[str]]]:
        """Train BERTopic model"""
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["calculate_probabilities"] = False


        # Function to read stopwords from a file
        def read_stop_words(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                custom_stopwords = file.read().splitlines()
                return custom_stopwords

        # Path to your custom stopwords file
        stop_words_file = "stopwords-nl.txt"

        # Read custom stopwords
        historical_stopwords = read_stop_words(stop_words_file)

        # Initialize CountVectorizer with custom stopwords
        dutch_stopwords = stopwords.words('dutch')
        english_stopwords = stopwords.words('english')
        # Combine the custom stopwords with the NLTK Dutch and English stopwords
        combined_stopwords = set(dutch_stopwords).union(set(english_stopwords)).union(set(historical_stopwords))
        vectorizer_model = CountVectorizer(stop_words=combined_stopwords)
        # from sklearn.cluster import KMeans
        # from sklearn.decomposition import PCA
        # # Initialize KMeans model
        # kmeans_model = KMeans(n_clusters=params.get("nr_topics", 10), random_state=42)
        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = BERTopic(vectorizer_model=vectorizer_model, **params)#, #hdbscan_model=kmeans_model, **params)

        start = time.time()
        topics, _ = model.fit_transform(data, self.embeddings)
        end = time.time()
        computation_time = float(end - start)

        bertopic_topics = []  # Initialize an empty list to store topics

        output_tm = {}  # Initialize an empty dictionary for output

        # Dynamic Topic Modeling
        if self.timestamps:
            topics_over_time = model.topics_over_time(
                data,
                topics,
                self.timestamps,
                nr_bins=self.nr_bins,
                evolution_tuning=False,
                global_tuning=False,
            )
            unique_timestamps = topics_over_time.Timestamp.unique()
            dtm_topics = {}
            for unique_timestamp in unique_timestamps:
                dtm_topic = topics_over_time.loc[
                    topics_over_time.Timestamp == unique_timestamp, :
                ].sort_values("Frequency", ascending=True)
                dtm_topic = dtm_topic.loc[dtm_topic.Topic != -1, :]
                dtm_topic = [topic.split(", ") for topic in dtm_topic.Words.values]
                dtm_topics[unique_timestamp] = {"topics": dtm_topic}

                all_words = [word for words in self.data.get_corpus() for word in words]

                updated_topics = []
                for topic in dtm_topic:
                    updated_topic = []
                    for word in topic:
                        if word not in all_words:
                            print(word)
                            updated_topic.append(all_words[0])
                        else:
                            updated_topic.append(word)
                    updated_topics.append(updated_topic)

                dtm_topics[unique_timestamp] = {"topics": updated_topics}
                bertopic_topics = updated_topics  # Update bertopic_topics with the dynamic topics

            output_tm = dtm_topics
            output_tm["topics"] = bertopic_topics  # Ensure 'topics' key is included
            
            
        if not self.timestamps:
            all_words = [word for words in self.data.get_corpus() for word in words]
            bertopic_topics = [
                [
                    vals[0] if vals[0] in all_words else all_words[0]
                    for vals in model.get_topic(i)[:10]
                ]
                for i in range(len(set(topics)) - 1)
            ]
            output_tm = {"topics": bertopic_topics}

        return output_tm, computation_time, bertopic_topics
    
    def evaluate(self, output_tm):
        """Using metrics and output of the topic model, evaluate the topic model"""
        if self.timestamps:
            results = {str(timestamp): {} for timestamp, _ in output_tm.items()}
            for timestamp, topics in output_tm.items():
                self.metrics = self.get_metrics()
                for scorers, _ in self.metrics:
                    for scorer, name in scorers:
                        score = scorer.score(topics)
                        results[str(timestamp)][name] = float(score)
        else:
            results = {}
            for scorers, _ in self.metrics:
                for scorer, name in scorers:
                    score = scorer.score(output_tm)
                    results[name] = float(score)

            if self.verbose:
                print("Results")
                print("============")
                for metric, score in results.items():
                    print(f"{metric}: {str(score)}")
                print(" ")

        return results

    def get_dataset(self):
        """Get dataset from OCTIS"""
        data = Dataset()

        if self.custom_dataset:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)
        return data

    def get_metrics(self):
        """Prepare evaluation measures using OCTIS"""
        npmi = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(npmi, "npmi")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))