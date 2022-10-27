import gc
import json
from tqdm import tqdm
import utility
import sqlite3 as sql

MAX_GRAM = 12 # Need to gen 12-15 grams


def calc_all_probs(gram_data):
    probs = {}

    for i in range(MAX_GRAM, 1, -1):
        probs[i] = utility.laplace_probs(gram_data.get(i), gram_data.get(i-1))
        print(sum(probs[i].values()))

    return probs


def combine_probs(all_probs):
    combined = {}

    for gram, prob in tqdm(all_probs.get(MAX_GRAM).items(), desc="Combining Probabilities: "):
        cur_prob = prob
        for i in range(MAX_GRAM-1, 0):
            cur_prob += all_probs[i][gram[:MAX_GRAM-i]]
        combined[gram] = cur_prob

    return combined


def check_freq_totals(all_probs):
    for prob in all_probs:
        if len(prob) < 4:
            print(prob, sum(all_probs[prob].values()))


def eval():
    data, hold_out = utility.read_file('sw-test.txt')

    entropy = utility.sw_evaluate(hold_out, 12)

    print(entropy)


def main():
    conn = sql.connect('sw-probs.db')
    print('Database Connection Established')

    data, hold_out = utility.read_file('sw-train.txt')

    conts = {}

    for i in range(1, MAX_GRAM + 1):
        if i == 1:
            continue
        ngrams = utility.gen_n_grams(data, n=i)

        sorted_freq = utility.sort_freqs_sw(ngrams, MAX_GRAM)
        if i == 2:
            conts = utility.get_continuations(sorted_freq)

        tmp_probs = utility.kneser_nay(sorted_freq, conts)

        cmd = 'INSERT INTO PROBS (`HISTORY`'
        vals = ' VALUES ('
        count = 0

        for history, prob_dict in tqdm(tmp_probs.items(), desc='Writing to DB'):
            cmd = 'INSERT INTO PROBS (`HISTORY`'
            vals = ' VALUES ('
            if '\'' in history:
                new_hist = ''
                for char in history:
                    if char == '\'':
                        new_hist += '\''

                    new_hist += char
                vals += '\'' + new_hist + '\''
            else:
                vals += '\'' + history + '\''
            for letter, prob in prob_dict.items():
                cmd += ',`' + letter + '`'
                vals += ',' + str(prob) + ''

            cmd += ')' + vals + ')'

            conn.execute(cmd)

        print("\nStoring on Disk\n")
        conn.commit()

        # try:
        #     old_probs = json.load(open('swahiliTEST.json'))
        #     all_probs = tmp_probs | old_probs
        #
        #     with open('swahiliTEST.json', 'w') as ofile:
        #         json.dump(all_probs, ofile)
        #
        #     del old_probs, all_probs
        #     gc.collect()
        # except json.decoder.JSONDecodeError:
        #     with open('swahiliTEST.json', 'w') as ofile:
        #         json.dump(tmp_probs, ofile)


    # del all_data
    # del sorted_freq['']
    # gc.collect()
    # # sorted_freq = utility.min_gram(sorted_freq, 1)
    #
    # conts = utility.get_continuations(sorted_freq)
    #
    # all_probs = utility.kneser_nay(sorted_freq, conts)
    #
    # del sorted_freq, conts
    # gc.collect()
    #
    # check_freq_totals(all_probs)
    # #
    # # utility.save_weights(all_probs, 'swahili.json')
    #
    # entropy = utility.evaluate(all_probs, hold_out, max_history=MAX_GRAM)
    #
    # print(entropy)


if __name__ == '__main__':
    eval()
