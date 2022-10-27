from tqdm import tqdm
import re
from math import log2
import json
import sqlite3 as sql


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789!\"'()-,.:;? "


def read_file(file_name):
    f = open(file_name, "r")

    full_text = f.read().split(" ")
    split_data = []

    for line in full_text: #tqdm(full_text, desc='Splitting words'):
        split_data.append(line.lower().strip())

    train = ' '.join(split_data[:int(len(split_data) * 0.8)])
    test = ' '.join(split_data[int(len(split_data) * 0.8):])

    return train, test


def read_test_file(file_name):
    f = open(file_name, "r")

    full_text = f.read().split(" ")
    split_data = []

    for line in full_text: #tqdm(full_text, desc='Splitting words'):
        split_data.append(line.lower().strip())

    return ' '.join(split_data)


def calc_freqs(n_grams):
    freqs = {}

    for ngram in tqdm(n_grams, desc='Calculating Frequencies: '):
        if ngram in freqs:
            freqs[ngram] += 1
        else:
            freqs[ngram] = 1

    return freqs


def sort_freqs_sw(ngrams, max_gram):
    sorted_freqs = {}

    for gram in tqdm(ngrams, desc='Sorting'):
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


def sum_db_rows(db_row):
    sum = 0
    for row in db_row:
        for entry in row[1:]:
            if isinstance(entry, float):
                sum += entry

    return sum

def sw_evaluate(text, max_history=5):
    conn_main = sql.connect('sw-probs.db')
    print('Loading DB into Memory')
    conn = sql.connect(':memory:')
    conn_main.backup(conn)
    conn.execute('CREATE INDEX Idx1 ON PROBS(HISTORY)')
    print('Database Connection Established')
    unigrams = conn.execute(
        'SELECT * FROM PROBS WHERE `HISTORY`=\'a\' OR`HISTORY`=\'b\' OR`HISTORY`=\'c\' OR`HISTORY`=\'d\' OR`HISTORY`=\'e\' OR`HISTORY`=\'f\' OR`HISTORY`=\'g\' OR`HISTORY`=\'h\' OR`HISTORY`=\'i\' OR`HISTORY`=\'j\' OR`HISTORY`=\'k\' OR`HISTORY`=\'l\' OR`HISTORY`=\'m\' OR`HISTORY`=\'n\' OR`HISTORY`=\'o\' OR`HISTORY`=\'p\' OR`HISTORY`=\'q\' OR`HISTORY`=\'r\' OR`HISTORY`=\'s\' OR`HISTORY`=\'t\' OR`HISTORY`=\'u\' OR`HISTORY`=\'v\' OR`HISTORY`=\'w\' OR`HISTORY`=\'x\' OR`HISTORY`=\'y\' OR`HISTORY`=\'z\' OR`HISTORY`=\'1\' OR`HISTORY`=\'2\' OR`HISTORY`=\'3\' OR`HISTORY`=\'4\' OR`HISTORY`=\'5\' OR`HISTORY`=\'6\' OR`HISTORY`=\'7\' OR`HISTORY`=\'8\' OR`HISTORY`=\'9\' OR`HISTORY`=\'0\' OR`HISTORY`=\'!\' OR`HISTORY`=\'?\' OR`HISTORY`=\'\'\'\' OR`HISTORY`=\'.\' OR`HISTORY`=\',\' OR`HISTORY`=\'(\' OR`HISTORY`=\')\' OR`HISTORY`=\':\' OR`HISTORY`=\';\' OR`HISTORY`=\' \' OR`HISTORY`=\'-\' OR`HISTORY`=\'"\'')

    init_prob = sum_db_rows(unigrams)

    history = []
    entropy = 0
    count = 0
    len_text = len(text)

    for cur_char in text: #tqdm(text, desc='Calculating Entropy'):
        count += 1
        hist_grams = history_to_grams(history)

        cur_prob = 0
        cmd = "SELECT * FROM PROBS WHERE `HISTORY`='" + cur_char + "'"

        if '\'' in cur_char:
            new_char = '\'\''
            cmd = "SELECT * FROM PROBS WHERE `HISTORY`='" + new_char + "'"

        cur_char_prob = conn.execute(cmd)

        cur_prob += sum_db_rows(cur_char_prob) / init_prob

        new_char = ''
        new_hist = ''

        for gram in hist_grams:
            new_char = ''
            new_hist = ''

            if len(gram) < 1:
                continue

            # if '\'' in cur_char:
            #     new_char = '\'\''
            # else:
            new_char = cur_char

            if '\'' in gram:
                new_hist = ''
                for char in history:
                    if char == '\'':
                        new_hist += '\''

                    new_hist += char
            else:
                new_hist = gram

            cmd = "SELECT `" + new_char + "` FROM PROBS WHERE `HISTORY`='" + new_hist + "'"
            prob = conn.execute(cmd)

            new_thing = prob.fetchall()

            if len(new_thing) < 1:
                continue

            try:
                cur_prob += ((1 / len(hist_grams)) * (new_thing[0][0]))
            except TypeError:
                continue

        entropy -= ((1 / len(text)) * log2(cur_prob))
        print('\rCurrent entropy: ', entropy, '\t', round(((count/len_text) * 100), 2), '% Done', end='', sep='')

        if len(history) >= max_history:
            history.pop(0)
        history.append(cur_char)

    print('\n')

    return entropy


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
            try:
                init_prob += sum(probs[char].values())
            except KeyError:
                continue

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


def load_weights(filename):
    data = {}
    with open(filename, 'r') as infile:
        data = json.load(infile)

    return data


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


def kneser_nay(sorted_grams, conts):
    l_probs = {}
    discount = 0

    for history in tqdm(sorted_grams, desc="Calculating Probabilities"):
        l_probs[history] = {}
        for char, count in sorted_grams[history].items():
            if count == 1:
                discount = 5
            if count == 2:
                discount = 0.00005
            else:
                discount = 0.00001

            numerator = max((count - discount), 0)

            interpolation = (discount / sum(sorted_grams[history].values()))

            prob = ((numerator / sum(sorted_grams[history].values())) + (interpolation * conts[char]))

            l_probs[history][char] = prob

    return l_probs
