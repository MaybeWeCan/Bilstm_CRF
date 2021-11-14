from Config import Config
from src.data_utils import CoNLLDataset, get_vocabs,\
    get_glove_vocab, write_vocab, load_vocab,\
    export_trimmed_glove_vectors


def main():
    """Procedure to build data
    1. It iterates over the whole dataset (train,dev and test)
    and extract the vocabularies in terms of words, tags, and
    characters.
    2. Having built the vocabularies it writes them in a file.
    The writing of vocabulary in a file assigns an id (the line #) to each word.
    3. It then extract the relevant GloVe vectors and stores them in a np array.
    such that the i-th entry corresponds to the i-th word in the vocabulary.
    Args:
        config: (instance of Config) has attributes like hyper-params...
    """
    # get config and processing of words
    config = Config()
    # processing_word = get_processing_word(lowercase=True)
    processing_word = None

    # 加载数据：words, tags
    dev = CoNLLDataset(config.filename_dev, processing_word)
    test = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab,
    vocab_words, vocab_tags = get_vocabs([train, dev, test])

    # 得到glove的vocab
    vocab_glove = get_glove_vocab(config.filename_glove)

    # 两个vocab求交集
    vocab = vocab_words & vocab_glove
    print("final len {}".format(len(vocab)))

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    # Saves glove vectors in numpy array
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                 config.filename_trimmed, config.embedding_dim)

    # # Build and save char vocab
    # train = CoNLLDataset(config.filename_train)
    # vocab_chars = get_char_vocab(train)
    # write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
