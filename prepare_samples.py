"""
Implementation of Skipgram with Negative Sampling
from Distributed Representations of Words and Phrases and their Compositionality
Mikolov (2013) - Not the thesis
https://arxiv.org/pdf/1310.4546.pdf

Built on Reuters dataset of articles
"""
import random
from tqdm import tqdm

from tensorflow.keras.datasets import reuters
import numpy as np

# SECTION 1
# Get corpus
vocab_size=10000

(train, _), (test, _) = reuters.load_data(path="reuters.npz",
                                                         num_words=vocab_size,
                                                         test_split=0.2,
                                                         seed=1337)

# A dictionary mapping words to an integer index
word_index = reuters.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

c = 8
K = 5

corpus = np.concatenate((train,test))
num_articles = corpus.shape[0]

# SECTION 2
# Gather unigram corpus statistics

# Per section 2.2 Negative sampling
# 1. Generate frequency table
# 2. Alpha smooth highest frequencies
# 3. get k noise words sampling from alpha smoothed distribution
# 4. Add k noise words as negative samples
freq = np.zeros((vocab_size,))
for i in tqdm(range(num_articles)):
    for w in corpus[i]:
        freq[w] += 1

raw_prob = freq / np.sum(freq)

alpha = 0.75
alpha_counts = np.power(freq, alpha)
alpha_counts[0:4] = 0 # Don't include symbols as noise words
alpha_prob = alpha_counts / np.sum(alpha_counts)


def get_noise_words(k):
    return np.argmax(np.random.multinomial(1, alpha_prob,k),axis=1)

# Per section 2.3 Subsampling of frequent words
t = 4.2e-6 # Goal seek such that smallest discard probability is ~0; originally 1e-5 in paper
discard_prob = 1. - np.sqrt(t * np.reciprocal(np.where(raw_prob == 0, 1., raw_prob)))

# SECTION 3
# Generate samples
ct_pairs = [] # Create list of target-context pair tuples
targets = [] # Create list of target; Set positive := 1, negative := 0


def add_example(t, c):
    """
    Add a positive example and K negative examples
    :param t: target word
    :param c: context word
    :return:
    """
    if t == c:
        return
    ct_pairs.append([t, c])
    targets.append(1)

    noise_words = get_noise_words(K)
    for noise in noise_words:
        if t == noise:
            noise = noise + 1
        ct_pairs.append([t, noise])
        targets.append(0)


for i in tqdm(range(num_articles)): # num_articles
    article = corpus[i]
    for j in range(len(article) - c - 1):
        w = article[j]
        if w < 4:
            continue # Skip symbols

        for k in range(1,c+1):
            v = article[j + k]
            if v < 4: # Skip symbols
                continue

            if random.random() > discard_prob[w]:
                add_example(w, v)

            if random.random() > discard_prob[v]:
                add_example(v, w)

pairs = np.array(ct_pairs)
targets = np.array(targets)

# for p in range(1000):
#     print(decode(ct_pairs[p]))
#     if (p+1) % 6 == 0:
#         print()

np.savez_compressed("sgns_samples.npz",pairs=pairs, targets=targets)


