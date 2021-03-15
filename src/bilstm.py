import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def get_glove_embeddings(vocab):
    '''
    Get pre-trained GloVe embeddings for the vocabulary from the training text.

    Parameters
    ----------
    vocab: <class 'set'>. The vocabulary from the training set.
    '''
    
    GLOVE_EMBEDDING = "/home/shuiyao_umass_edu/nlp_data/glove/glove.6B.50d.txt"

    PAD_TOKEN = 0

    word2idx = {'PAD': PAD_TOKEN}
    weights = []
    index = 0
    
    with open(GLOVE_EMBEDDING, 'r') as file:
        for i, line in enumerate(file):
            spt = line.split()
            word = spt[0]
            if(word in vocab):
                word_weights = np.asarray(spt[1:], dtype=np.float32)
                word2idx[word] = index + 1 # PAD is our zeroth index so shift by one
                weights.append(word_weights)
                index += 1

    EMBEDDING_DIMENSION = len(weights[0])
    # Insert the PAD weights at index 0 now we know the embedding dimension
    weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))

    # Append unknown and pad to end of vocab and initialize as random
    UNKNOWN_TOKEN=len(weights)
    word2idx['UNK'] = UNKNOWN_TOKEN
    weights.append(np.random.randn(EMBEDDING_DIMENSION))

    # Construct our final vocab
    weights = np.asarray(weights, dtype=np.float32)

    # VOCAB_SIZE=weights.shape[0]
    # weights: (VOCAB_SIZE, 50)
    return word2idx, weights

def build_classifier(text):
    MAX_VOCAB_SIZE = 20000
    encoder = TextVectorization(max_tokens=MAX_VOCAB_SIZE)
    encoder.adapt(text)
    vocabset = set(encoder.get_vocabulary())
    vocab_size = len(encoder.get_vocabulary())

    word2idx, weights = get_glove_embeddings(vocabset)
    embedding_matrix = np.zeros((vocab_size, weights.shape[1]))
    for i, word in enumerate(encoder.get_vocabulary()):
        vec = word2idx.get(word)
        if(vec is not None):
            embedding_matrix[i] = weights[vec]
    
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim = embedding_matrix.shape[0],
            output_dim = embedding_matrix.shape[1],
            weights = embedding_matrix, 
            mask_zero=True,
            trainable=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

