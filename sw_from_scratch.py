from tqdm import tqdm
import utility
import gc

MAX_GRAM = 10


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
    data, hold_out = utility.read_file('sw-train.txt')

    all_data = {}

    for i in range(1, MAX_GRAM+1):
        ngrams = utility.gen_n_grams(data, n=i)

        all_data[i] = ngrams

    del data
    gc.collect()

    sorted_freq = utility.sort_freqs(all_data, MAX_GRAM)
    del all_data
    del sorted_freq['']
    gc.collect()
    # sorted_freq = utility.min_gram(sorted_freq, 1)

    conts = utility.get_continuations(sorted_freq)

    all_probs = utility.kneser_nay(sorted_freq, conts)

    del sorted_freq, conts
    gc.collect()

    check_freq_totals(all_probs)

    utility.save_weights(all_probs, 'swahili.json')

    entropy = utility.evaluate(all_probs, hold_out, max_history=MAX_GRAM)

    print(entropy)


if __name__ == '__main__':
    main()
