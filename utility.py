from tqdm import tqdm
import re
import math
import json

def read_file(file_name):
    f = open(file_name, "r")

    full_text = f.read().split(" ")
    split_data = []

    for line in tqdm(full_text, desc='Splitting words'):
        split_data.append(line.lower())

    return ' '.join(split_data)


def calc_freqs(n_grams):
    freqs = {}

    for ngram in tqdm(n_grams, desc='Calculating Frequencies: '):
        if ngram in freqs:
            freqs[ngram] += 1
        else:
            freqs[ngram] = 1

    return freqs


def sort_freqs(ngrams, max_gram):
    sorted_freqs = {}

    for i in tqdm(range(1, max_gram+1), desc='Sorting Frequencies:'):
        for gram in ngrams[i]:
            history = gram[:-1]
            letter = gram[-1]
            if len(letter) < 1:
                continue

            if history in sorted_freqs:
                if letter in sorted_freqs[history]:
                    sorted_freqs[history][letter] += 1
                else:
                    sorted_freqs[history][letter] = 1
            else:
                sorted_freqs[history] = {letter: 1}

    return sorted_freqs


def min_gram(ngrams, min=5):
    cleaned_grams = {}

    for key in ngrams:
        unk = 0
        cleaned_grams[key] = {}
        for letter, count in ngrams[key].items():
            if count >= min:
                cleaned_grams[key][letter] = count
            else:
                unk += count

        cleaned_grams[key]['UNK'] = unk

    return cleaned_grams

def gen_n_grams(data, n=3):
    descript = "Generating " + str(n) + " Grams:"

    n_grams = [''.join(data[i:i+n]) for i in tqdm(range(len(data) - n + 1), desc=descript)]

    return n_grams


def calc_entropy(probs):
    entropy = 0

    for prob in tqdm(probs.values(), desc="Calculating Entropy: "):
        entropy -= (prob * math.log(prob, 2))

    return entropy


def laplace_probs(sorted_grams, alpha=0.1):
    # l_probs = {}
    # vocab = sum(ngrams.values())
    #
    # print('\'', max(ngrams, key=ngrams.get), '\' \'',  max(n1grams, key=n1grams.get), '\'', sep='')
    #
    # for ngram, val in tqdm(ngrams.items(), desc='Calculating Probabilities: '):
    #     prob = (val + alpha) / (n1grams[ngram[:-1]] + (alpha * vocab))
    #     l_probs[ngram] = prob
    #
    # return l_probs

    l_probs = {}
    for history in tqdm(sorted_grams, desc="Calculating Probabilities"):
        vocab = sum(sorted_grams[history].values())
        l_probs[history] = {}
        for char, count in sorted_grams[history].items():
            try:
                prob = (count) / (sum(sorted_grams[history].values()))
            except KeyError:
                prob = (count + alpha) / (sum(sorted_grams[history].values()) + (vocab * alpha))

            l_probs[history][char] = prob

    return l_probs

def save_weights(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
