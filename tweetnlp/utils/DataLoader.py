import os

from .tweetokenize.tokenizer import Tokenizer

from collections import Counter
import pandas as pd
import numpy as np
import pickle

import keras
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataLoader:

    dataset, wv_dict = None, None
    task = 'classification'
    maxlen = 50
    emb_dim = 5
    seed = 1337

    def __init__(self, dataset_path=None, wv_path=os.path.join(BASE_DIR, 'pickles', 'pinoy_wv_dict_05.pickle'),
                 w2v_model=None, task='classification', has_label=True, dim=None):
        """Initialize the DataLoader

        Arguments:
            dataset_path {str} -- location of the dataset
            wv_path {str} -- location of the word vector dictionary
            w2v_model {gensim word2vec} -- load a trained word2vec model to be used as wv_dict
        """
        # Load dataset
        if type(dataset_path) == str:
            dataset_path = dataset_path.encode(
                'unicode-escape').decode().replace('\\\\', '\\')  # Fix windows path

            if has_label == True:
                self.dataset = pd.read_csv(
                    dataset_path, sep='\t', names=['y', 'x'])
            else:
                self.dataset = pd.read_csv(
                    dataset_path, sep='\t', names=['x'])

        # Load word vector dictionary
        if type(wv_path) == str and has_label == True:
            if w2v_model == None:
                with open(wv_path, 'rb') as file:
                    self.wv_dict = pickle.load(file)
            else:
                self.create_dict_from_w2v(w2v_model)
            self.get_embedding_weights()

        # Set the task to perform (either classification or regression)
        self.task = task

        # Set has_label
        self.has_label = has_label

        # Reset the class_dict
        if self.task == 'classification' and has_label == True:
            self.create_class_dict()
        elif self.task == 'regression' and has_label == True:
            self.class_dict = None
            self.dataset_summary = self.dataset.describe()

        # Reduce dimensionality
        if dim:
            self.reduce_dimension(dim=dim)

    def set_dataset(self, dataset, task='classification'):
        """Input an already loaded dataset

        Arguments:
            dataset {DataFrame} -- a pandas dataframe with columns 'x' for the tweets and 'y' for the labels
            task {str} -- set the task to either classification or regression
        """
        self.dataset = dataset

        # Set task
        self.set_task(task)

        if self.task == 'classification':
            # Reset the class_dict
            self.create_class_dict()

            # Summarize dataset
            self.summarize_dataset()

        elif self.task == 'regression':
            self.class_dict = None
            self.dataset_summary = self.dataset.describe()

    def set_wv_dict(self, wv_dict):
        """Input an already loaded wv_dict

        Arguments:
            wv_dict {dict} -- a dictionary where keys are the vocabulary words and values are the vector representations
        """
        self.wv_dict = wv_dict

    def create_dict_from_w2v(self, w2v_model):
        """Create a word vector dictionary from word2vec model

        Arguments:
            w2v_model {word2vec} -- a gensim word2vec model
        """
        words, vectors = [], []
        for key in sorted(w2v_model.wv.vocab.keys()):
            words.append(key)
            vectors.append(w2v_model.wv[key])

        self.wv_dict = {words[i]: vectors[i] for i in range(len(words))}

    def save_wv_dict(self, filename, save_type='pickle'):
        """Allow saving of created word2vec dictionary

        Arguments:
            filename {str} -- filename of output file
            save_type {str} -- use 'pickle' to save as pickle; else, use 'list' to save as a sorted list
        """
        with open(filename, 'wb') as f:
            if save_type == 'pickle':
                pickle.dump(self.wv_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif save_type == 'list':
                for idx, key in enumerate(sorted(self.wv_dict.keys)):
                    data = str(idx + 2) + '\t' + key + '\n'
                    f.write(data.encode())

    def reduce_dimension(self, algorithm=PCA, dim=5, seed=1337):
        """Reduce the dimension of the loaded wv_dict

        Arguments:
            algorithm {dimensionality reduction algorithm} -- a dimesionality reduction algorithm with a fit_transform function. Set to sklearn's PCA by default.
            dim {int} -- the desired dimension for the wv_dict
            seed {int} -- a seed value used for the algorithm's random_state
        """

        word_vectors = []
        for key in sorted(self.wv_dict.keys()):
            word_vectors.append(self.wv_dict[key])

        algo = algorithm(n_components=dim, random_state=seed)
        reduced_wv = algo.fit_transform(word_vectors)

        for idx, key in enumerate(sorted(self.wv_dict.keys())):
            self.wv_dict[key] = reduced_wv[idx]

    def get_embedding_weights(self, emb_dim=None):
        """Create embedding weights for transfer learning

        Arguments:
            emb_dim {int} -- the number of dimension the embedding has
        """

        # Get embedding dimension
        if emb_dim == None:
            # Infer dimension from wv_dict
            key_1 = list(self.wv_dict.keys())[0]
            self.emb_dim = self.wv_dict[key_1].shape[0]
        else:
            self.emb_dim = emb_dim

        embedding_weights = np.zeros((len(self.wv_dict) + 2, self.emb_dim))
        for idx, word in enumerate(sorted(self.wv_dict.keys())):
            try:
                embedding_weights[idx + 2, :] = self.wv_dict[word]
            except:
                pass

        # Create embeddings for padding and unkown words
        embedding_weights[0, :] = np.zeros(self.emb_dim)
        embedding_weights[1, :] = np.ones(self.emb_dim)

        return embedding_weights

    def get_class_weights(self, smooth_factor=0.1):
        """Returns the weights for each class based on the frequencies of the samples

        Arguments:
            :param smooth_factor: factor that smooths extremely uneven weights
            :param y: list of true labels (the labels must be hashable)
        Returns:
            dict -- dictionary with the weight for each class
        """

        counter = Counter(self.dataset['y'])

        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in sorted(counter.keys()):
                counter[k] += p

        majority = max(counter.values())
        class_weights = {idx: float(
            majority / counter[cls]) for idx, cls in enumerate(sorted(counter.keys()))}

        return class_weights

    def set_task(self, task):
        """Change the task (classification|regression)

        Arguments:
            task {str} -- set the task to either classification or regression
        """
        self.task = task

    def create_class_dict(self):
        """Create a dictionary of the classes and their numeric representation"""
        self.class_dict = {label: idx for idx,
                           label in enumerate(sorted(set(self.dataset['y'])))}

    def summarize_dataset(self):
        """Create a summary of the loaded dataset"""

        dataset_summary = dict(Counter(self.dataset['y']))
        dataset_summary = pd.DataFrame(
            (dataset_summary.keys(), dataset_summary.values())).T
        dataset_summary.columns = ['label', 'frequency']
        dataset_summary.sort_values(by=['label'], inplace=True)
        dataset_summary.reset_index(drop=True, inplace=True)

        self.dataset_summary = dataset_summary
        return self.dataset_summary

    def tokenize_dataset(self, text_data, verbose=False, **kwargs):
        """Convert the list of text data into word tokens

        Arguments:
            text_data {list} -- list of texts like tweets

        Returns:
            list -- list containing lists of word tokens
        """

        tokenizer = Tokenizer(strict=False, **kwargs)
        return tokenizer.tokenize_set(text_data, verbose=verbose)

    def vectorize_x(self, input_tokens, wv_dict, maxlen=50):
        """Convert the list of input tokens into numerical vectors
           0  -- padding
           1  -- unknown words
           2+ -- words in vocabulary

        Arguments:
            input_tokens {list} -- list containing lists of word tokens
            word_vector_path {dict} -- a dictionary of word vectors where the words are keys and the vectors are the values

        Returns:
            list -- list containing lists of indexes
        """

        word_index = {key: idx + 2 for idx,
                      key in enumerate(sorted(wv_dict.keys()))}

        input_vectors = []
        for word_tokens in input_tokens:
            index_tokens = []
            for token in word_tokens:
                try:
                    index_tokens.append(word_index[token])
                except:
                    index_tokens.append(1)
            input_vectors.append(index_tokens)

        return keras.preprocessing.sequence.pad_sequences(input_vectors, maxlen=maxlen)

    def categorize_y(self, labels, class_dict):
        """Convert the labels into categorical format

        Arguments:
            labels {Series} -- the labels
            class_dict {dict} -- a dictionary with the classes as keys and their index as values

        Returns:
            ndarray -- array of categorical representation
        """
        labels = [class_dict[i] for i in labels]
        return keras.utils.to_categorical(labels, len(class_dict))

    def load_data(self, maxlen=50, verbose=False, x_cols=[], **kwargs):
        """Create train and test datasets that can be readily used by machine learning classifiers

        Arguments:
            x_cols = list of column names to include as input
            maxlen {int} -- the dimension of the vectors representing the input text data (tweets)
            verbose {boolean} -- a dictionary with the classes as keys and their index as values

        Returns:
            ndarray -- array of categorical representation
        """

        # Tokenize the input data
        x = self.tokenize_dataset(self.dataset['x'], verbose=verbose)

        if self.task == 'classification' and self.has_label == True:
            # Create dataset summary
            self.summarize_dataset()

            # Vectorize the input data
            self.maxlen = maxlen
            x = self.vectorize_x(x, self.wv_dict, maxlen=self.maxlen)

            # Categorize the labels
            y = self.categorize_y(self.dataset['y'], self.class_dict)

            # Stratify
            stratify = y

        elif self.task == 'regression' and self.has_label == True:
            y = self.dataset[x_cols]
            # Don't stratify
            stratify = None

        elif self.has_label == False:
            # Vectorize the input data
            self.maxlen = maxlen
            x = self.vectorize_x(x, self.wv_dict, maxlen=self.maxlen)

        # Create train and test datasets
        if self.has_label == True:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                stratify=stratify,
                                                                random_state=self.seed,
                                                                **kwargs)
        else:
            x_train, x_test = train_test_split(
                x, random_state=self.seed, **kwargs)
            y_train, y_test = None, None

        return x_train, x_test, y_train, y_test
