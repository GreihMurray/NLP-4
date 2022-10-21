from tqdm import tqdm
import utility
import gc
import sqlite3 as sql

MAX_GRAM = 15
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789!\"'()-,.:;? "

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



def interpolate(cwe):
    conn_main = sql.connect('sw-probs.db')
    print('Loading DB into Memory')
    conn = sql.connect(':memory:')
    conn_main.backup(conn)
    conn.execute('CREATE INDEX Idx1 ON PROBS(HISTORY)')
    print('Database Connection Established')

    sw_weight = 0.0
    cwe_weight = 1.0

    for history, probs in tqdm(cwe.items(), desc='Interpolating Weights'):
        if len(history) > 12:
            continue

        if len(history) < 1:
            continue

        new_hist = ''

        if '\'' in history:
            for char in history:
                if char == '\'':
                    new_hist += '\''
                new_hist += char
        else:
            new_hist = history

        #print("`", history, "`", "`", new_hist, "`", sep='')

        cmd = 'SELECT * FROM PROBS WHERE `HISTORY`=\'' + new_hist + '\''
        swp = conn.execute(cmd)
        swp = swp.fetchall()

        if len(swp) <= 0:
            continue

        swp = swp[0][1:]
        sw_d = {}

        for i in range(0, len(ALPHABET)):
            if isinstance(swp[i], float):
                sw_d[ALPHABET[i]] = swp[i]

        #
        # cmd += new_hist
        #
        # swp = conn.execute(cmd)

        shared = sw_d.keys() | probs.keys()

        for key in shared:
            if key in sw_d.keys() and key in probs.keys():
                cwe[history][key] = (cwe[history][key] * cwe_weight) + (sw_d[key] * sw_weight)
            elif key in sw_d.keys() and key not in probs.keys():
                cwe[history][key] = sw_d[key] * sw_weight
            elif key in probs.keys() and key not in sw_d.keys():
                cwe[history][key] = cwe[history][key] * cwe_weight

    return cwe


def main():
    data, hold_out = utility.read_file('cwe-train.txt')

    all_data = {}

    for i in range(1, MAX_GRAM+1):
        ngrams = utility.gen_n_grams(data, n=i)

        all_data[i] = ngrams

    del data

    sorted_freq = utility.sort_freqs(all_data, MAX_GRAM)
    del all_data
    del sorted_freq['']
    gc.collect()
    # sorted_freq = utility.min_gram(sorted_freq, 1)

    conts = utility.get_continuations(sorted_freq)

    all_probs = utility.kneser_nay(sorted_freq, conts)

    del sorted_freq, conts
    gc.collect()

    #all_probs = interpolate(all_probs)

    check_freq_totals(all_probs)

    utility.save_weights(all_probs, 'kwere.json')

    entropy = utility.evaluate(all_probs, hold_out, max_history=MAX_GRAM)

    print(entropy)


if __name__ == '__main__':
    main()
