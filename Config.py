class Config():

    # general config
    dir_output = "results/test/"
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    # glove files
    filename_glove = "data/word2vec.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/word2vec.txt.trimmed.npz"
    use_pretrained = True

    # dataset
    filename_dev = "data/data_dev.txt"
    filename_test = "data/data_test.txt"
    filename_train = "data/data_train.txt"

    # vocab
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # embeddings
    embedding_dim = 128

    # training
    vocab_size = -1
    label_size = -1

    train_embeddings = False
    nepochs = 20
    dropout = 0.4
    batch_size = 32
    lr_method = "adam"
    lr = 0.002            #0.001
    lr_decay = 0.9
    clip = 5  # if negative, no clipping
    save_dir = 'save_model'

    steps_check = 50
    require_improve = 10000

    # model hyperparameters
    lstm_hidden_size = 256 # lstm on chars
    lstm_num_layers = 1


    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = False  # if char embedding, training is 3.5x slower on CPU
