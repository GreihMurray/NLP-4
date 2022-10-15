from tqdm import tqdm
import re
from math import log2
import json


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789!\"'()-,.:;? "


def read_file(file_name):
    f = open(file_name, "r")

    full_text = f.read().split(" ")
    split_data = []

    for line in tqdm(full_text, desc='Splitting words'):
        split_data.append(line.lower().strip())

    train = ' '.join(split_data[:int(len(split_data) * 0.8)])
    test = ' '.join(split_data[int(len(split_data) * 0.8):])

    return train, test


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
        cleaned_grams[key]['UNK'] = 1

    return cleaned_grams

def gen_n_grams(data, n=3):
    descript = "Generating " + str(n) + " Grams:"

    n_grams = [''.join(data[i:i+n]) for i in tqdm(range(len(data) - n + 1), desc=descript)]

    return n_grams


def history_to_grams(history):
    grams = []

    for i in range(len(history), 0, -1):
        grams.append(''.join(history[i:]))

    return grams



def evaluate(probs, text, max_history=5):
    history = []
    entropy = 0
    count = 0

    for cur_char in tqdm(text, desc='Calculating Entropy'):
        count += 1
        hist_grams = history_to_grams(history)

        cur_prob = 0
        init_prob = 0

        for char in ALPHABET:
            init_prob += sum(probs[char].values())

        cur_prob += sum(probs[cur_char].values()) / init_prob

        for gram in hist_grams:
            try:
                cur_prob += ((1 / len(hist_grams)) * (probs[gram][cur_char]))
            except KeyError:
                continue

        entropy -= ((1 / len(text)) * log2(cur_prob))

        if len(history) >= max_history:
            history.pop(0)
        history.append(cur_char)

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


def get_continuations(ngrams):
    conts = {}

    for history in tqdm(ngrams, desc='Counting Continuations'):
        if len(history) > 1:
            continue
        for letter in ALPHABET:
            if letter in ngrams[history].keys():
                if letter in conts:
                    conts[letter] += 1
                else:
                    conts[letter] = 1

    return conts


def kneser_nay(sorted_grams, conts, discount=0.000001):
    l_probs = {}

    for history in tqdm(sorted_grams, desc="Calculating Probabilities"):
        l_probs[history] = {}
        for char, count in sorted_grams[history].items():
            numerator = max((count - discount), 0)

            interpolation = (discount / sum(sorted_grams[history].values()))

            prob = ((numerator / sum(sorted_grams[history].values())) + (interpolation * conts[char]))

            l_probs[history][char] = prob

    return l_probs
