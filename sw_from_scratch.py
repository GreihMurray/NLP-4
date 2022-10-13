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


def main():
    data = utility.read_file('sw-train.txt')

    all_data = {}

    for i in range(1, MAX_GRAM+1):
        ngrams = utility.gen_n_grams(data, n=i)
        #freqs = utility.calc_freqs(ngrams)

        all_data[i] = ngrams

    print(all_data.keys())

    sorted_freq = utility.sort_freqs(all_data)

    print('C', sorted_freq['c'])
    print(min(sorted_freq['c'], key=sorted_freq['c']), sorted_freq[min(sorted_freq['c'], key=sorted_freq['c'])])

    # all_probs = calc_all_probs(all_data)
    # probs = combine_probs(all_probs)
    #
    # entropy = utility.calc_entropy(probs)
    #
    # print(entropy)


if __name__ == '__main__':
    main()
