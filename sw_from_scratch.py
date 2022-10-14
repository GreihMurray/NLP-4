from collections import Counter
from tqdm import tqdm
import utility

MAX_GRAM = 5


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


def main():
    data = utility.read_file('sw-train.txt')

    all_data = {}

    for i in range(1, MAX_GRAM+1):
        ngrams = utility.gen_n_grams(data, n=i)

        all_data[i] = ngrams

    sorted_freq = utility.sort_freqs(all_data, MAX_GRAM)

    del sorted_freq['']

    conts = utility.get_continuations(sorted_freq)

    all_probs = utility.kneser_nay(sorted_freq, conts)

    check_freq_totals(all_probs)

    utility.save_weights(all_probs, 'swahili.json')



if __name__ == '__main__':
    main()
