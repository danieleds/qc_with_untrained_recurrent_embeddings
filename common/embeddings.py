import torch
import gensim
import timeit
import os
from common.Datasets import download_and_extract


def get_word_embedding(word, embeddings):
    if word in embeddings:
        return torch.from_numpy(embeddings[word])
    elif word.lower() in embeddings:
        return torch.from_numpy(embeddings[word.lower()])
    elif word == '-LRB-':
        return torch.from_numpy(embeddings['('])
    elif word == '-RRB-':
        return torch.from_numpy(embeddings[')'])
    elif word == '-LCB-':
        return torch.from_numpy(embeddings['{'])
    elif word == '-RCB-':
        return torch.from_numpy(embeddings['}'])
    elif word == "''":
        return torch.from_numpy(embeddings["'"])
    elif word == '`' or word == '``':
        return torch.from_numpy(embeddings["'"])
    elif word == '\\/':
        return torch.from_numpy(embeddings['/'])
    else:
        return None


def get_labels(t, embeddings):
    """

    :param t: list of tokens
    :param embeddings: fasttext embeddings
    :return: a tensor (n_tokens, dim_embedding)
    """
    labels = torch.zeros((len(t), embeddings.vector_size))
    for i in range(len(t)):
        lbl = get_word_embedding(t[i], embeddings)
        if lbl is None:
            lbl = torch.rand(embeddings.vector_size)
        labels[i, :] = lbl
    return labels


def load_fasttext_embeddings(lang="en", silent=False, root='.data'):
    embeddings_path = download_and_extract(
        [f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz'],
        root=root,
        main_folder_name=f'fasttext_embeddings/{lang}',
        check=os.path.join(root, f'fasttext_embeddings/{lang}')
    )
    start_time = timeit.default_timer()
    embeddings = gensim.models.fasttext.load_facebook_vectors(os.path.join(embeddings_path, f'cc.{lang}.300.bin'))
    elapsed = timeit.default_timer() - start_time
    if not silent:
        print(f"Embeddings loaded in {round(elapsed)} seconds.")
    return embeddings
