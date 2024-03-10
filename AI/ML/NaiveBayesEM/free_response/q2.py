import numpy as np
import pandas as pd

from free_response.data import build_dataset
from src.naive_bayes_em import NaiveBayesEM


def main():
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    width = max(map(len, vocab))

    nb = NaiveBayesEM(max_iter=1)
    nb.fit(data, labels)

    f_scores = nb.beta[:, 1] / nb.beta[:, 0]
    f_idxs = np.argsort(f_scores)
    # print(f_idxs)
    
    headers = ["Word", "Beta[:, 0]", "Beta[:, 1]"]
    words, prob0, prob1 = [], [], []
    n_to_print = 3
    # print(type(vocab))
    # print(vocab)
    for idx in f_idxs[:n_to_print]:
        # print('idx:', idx)
        # print(type(vocab[idx]))
        words.append(f"{vocab[idx]:{width}s}")
        prob0.append(f"{nb.beta[idx, 0]:.3f}")
        prob1.append(f"{nb.beta[idx, 1]:.3f}")

    for idx in reversed(f_idxs[-n_to_print:]):
        words.append(f"{vocab[idx]:{width}s}")
        prob0.append(f"{nb.beta[idx, 0]:.3f}")
        prob1.append(f"{nb.beta[idx, 1]:.3f}")

    closest_to_one = np.argsort(np.max(
        np.stack([f_scores, 1 / f_scores], axis=1), axis=1))
    for idx in closest_to_one[:n_to_print]:
        words.append(f"{vocab[idx]:{width}s}")
        prob0.append(f"{nb.beta[idx, 0]:.3f}")
        prob1.append(f"{nb.beta[idx, 1]:.3f}")

    df = pd.DataFrame(dict(zip(headers, [words, prob0, prob1])))
    print(df.to_string(justify='left', index=False))


if __name__ == "__main__":
    main()
