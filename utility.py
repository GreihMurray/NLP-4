from tqdm import tqdm
import re
import math

# TODO
    # Remove non alphabetical characters
def read_file(file_name):
    f = open(file_name, "r")

    full_text = f.read().split(" ")
    split_data = []

    for line in tqdm(full_text, desc='Splitting words'):
        split_data.append(re.sub('[^a-zA-Z]', '', line).lower())

    return ' '.join(split_data)


def calc_freqs(n_grams):
    freqs = {}

    for ngram in tqdm(n_grams, desc='Calculating Frequencies: '):
        if ngram in freqs:
            freqs[ngram] += 1
        else:
            freqs[ngram] = 1

    return freqs


def sort_freqs(freqs):
    sorted_freqs = {}
    max_gram = max(freqs, key=freqs.get)

    for i in tqdm(range(1, max_gram), desc='Sorting Frequencies:'):
        for gram in freqs[i]:
            history = gram[:-1]
            letter = gram[-1]

            if letter in sorted_freqs:
                if history in sorted_freqs[letter]:
                    sorted_freqs[letter][history] += 1
                else:
                    sorted_freqs[letter][history] = 1
            else:
                sorted_freqs[letter] = {gram: 1}

    return sorted_freqs


def gen_n_grams(data, n=3):
    descript = "Generating " + str(n) + " Grams:"

    n_grams = [''.join(data[i:i+n]) for i in tqdm(range(len(data) - n + 1), desc=descript)]

    return n_grams


def calc_entropy(probs):
    entropy = 0

    for prob in tqdm(probs.values(), desc="Calculating Entropy: "):
        entropy -= (prob * math.log(prob, 2))

    return entropy


def laplace_probs(ngrams, n1grams, alpha=0.1):
    l_probs = {}
    vocab = sum(ngrams.values())

    print('\'', max(ngrams, key=ngrams.get), '\' \'',  max(n1grams, key=n1grams.get), '\'', sep='')

    for ngram, val in tqdm(ngrams.items(), desc='Calculating Probabilities: '):
        prob = (val + alpha) / (n1grams[ngram[:-1]] + (alpha * vocab))
        l_probs[ngram] = prob

    return l_probs
