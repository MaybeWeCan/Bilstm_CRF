import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

from dataloader import RMRB_Dataset, RMRB_collate_fn
from Config import Config
from torch.utils.data import DataLoader
from src.BiLstm_CRF import Bilstm_CRF
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score

UNK = "[UNK]"
PAD = "[PAD]"

""" 设置随机数种子 """


def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_vocab(filename, is_vocab = True):
    vocabs = []
    if is_vocab:
        vocabs.append(PAD)
        vocabs.append(UNK)
    with open(filename, encoding='utf-8') as f:
        for word in f:
            word = word.strip()
            vocabs.append(word)
    return vocabs

def vocab_id_dict(vocabs):

    id_to_word = {}
    word_to_id = {}
    for idx, word in enumerate(vocabs):
        id_to_word[idx] = word
        word_to_id[word] = idx
    return word_to_id, id_to_word

def get_logger(name, path='train.log', level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    f_handler = logging.FileHandler(path)
    f_handler.setLevel(level)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(handler)
    logger.addHandler(f_handler)

    return logger


def evaluate_f1_loss(model, config, dev_dataloader, id_to_word, id_to_tags,device):
    with torch.no_grad():

        total_loss = 0
        results = []
        total_f1_score = 0
        f1_count = 0
        for batch in dev_dataloader:

            batch_seqs, batch_tags, batch_mask = batch
            batch_seqs, batch_tags, batch_mask = batch_seqs.to(device), batch_tags.to(device), batch_mask.to(device)

            # 没传入tags
            batch_paths = model(batch_seqs, batch_mask)
            loss = model(batch_seqs, batch_mask, batch_tags)
            total_loss += loss.item()

            """ 忽略<pad>标签，计算每个样本的真实长度 """
            lengths = [len([j for j in i if j > 0]) for i in batch_mask.tolist()]

            tag_ids = batch_tags.tolist()

            for i in range(len(batch_seqs)):
                result = []
                # string = batch_seqs[i][:lengths[i]]
                string = []
                for char_ in batch_seqs[i][:lengths[i]]:
                    string.append(id_to_word[int(char_)])

                f_score = f1_score(tag_ids[i][:lengths[i]],batch_paths[i][:lengths[i]], average='micro')
                total_f1_score += f_score
                f1_count += 1

        aver_loss = total_loss / (len(dev_dataloader) * config.batch_size)
        aver_f1 = total_f1_score / f1_count
        return aver_loss, aver_f1


def evaluate(model, config, dev_dataloader, id_to_word, id_to_tags, logger, device, test=False):

    """ 得到预测的标签（非id）和损失 """
    aver_loss,f1 = evaluate_f1_loss(model, config, dev_dataloader, id_to_word, id_to_tags, device)
    return f1, aver_loss


def train(name,device):
    config = Config()
    logger = get_logger(name, path='train.log', level=logging.INFO)

    # Read Vocab
    logger.info("Read the vocab")
    vocabs = load_vocab(config.filename_words)
    config.vocab_size = len(vocabs)
    #tags = load_vocab(config.filename_tags, is_vocab = False)
    tags = {'O': 0, 'B-LOC': 1,  'I-LOC': 2, 'B-PER': 3, 'I-PER': 4,'B-ORG': 5,  'I-ORG': 6}
    config.label_size = len(tags)

    vocabs_to_id, id_to_vocabs = vocab_id_dict(vocabs)
    tags_to_id, id_to_tags = vocab_id_dict(tags)

    logger.info("tags_to_id: {}".format(tags_to_id))
    # logger.info("vocab_to_id: {}".format(vocabs_to_id))

    # Create Dataset
    logger.info("Create Dataset")
    train_dataset = RMRB_Dataset(config.filename_train, vocabs_to_id, tags_to_id)
    dev_dataset = RMRB_Dataset(config.filename_dev , vocabs_to_id, tags_to_id)

    logger.info('train_dataset[1]: {}'.format(train_dataset[1]))

    # Create Dadaloader, Pad & mask
    logger.info("Create Dadaloader, Pad & mask")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers = 4, collate_fn=RMRB_collate_fn)
    logger.info("It is all {}".format(len(train_dataset)))

    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers = 4, collate_fn=RMRB_collate_fn)

    # model
    logger.info("Create model")
    model = Bilstm_CRF(config,is_layernorm=True)
    
    # 加载转化后的文件
    wvmodel = KeyedVectors.load_word2vec_format(config.filename_glove)
    weight = torch.zeros(config.vocab_size, config.embedding_dim)

    for i in range(len(wvmodel.vocab)):
        try:
            index = vocabs_to_id[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(id_to_vocabs[vocabs_to_id[wvmodel.index2word[i]]]))

    logger.info("Load pretrained embedding")
    model.vocab_embedding.weight.data.copy_(weight)

    model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.lr_decay)

    # train & dev
    logger.info("Start Training....")

    total_batch = 0
    dev_best_f1 = float('-inf')
    last_improve = 0
    flag = False

    for epoch in range(config.nepochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.nepochs))

        for index, batch in enumerate(train_dataloader):

            model.train()

            optimizer.zero_grad()
            batch_seqs, batch_tags, batch_mask = batch
            batch_seqs, batch_tags, batch_mask = batch_seqs.to(device), batch_tags.to(device), batch_mask.to(device)

            if total_batch == 0:
                logger.info("Batch_seq {} ".format(batch_seqs))
                logger.info("batch_tags {} ".format(batch_tags))
                logger.info("batch_mask {} ".format(batch_mask))

                logger.info("Batch_seq_one {} ".format(batch_seqs[0]))
                logger.info("batch_tags_one {} ".format(batch_tags[0]))
                logger.info("batch_mask_one {} ".format(batch_mask[0]))

            loss = model(batch_seqs, batch_mask, batch_tags)
            loss.backward()

            """ 梯度截断，最大梯度为5 """
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip)
            optimizer.step()

            if total_batch % config.steps_check == 0:

                model.eval()

                dev_f1, dev_loss = evaluate(model, config, dev_dataloader, id_to_vocabs, id_to_tags, logger, device)

                """ 以f1作为early stop的监控指标 """
                if dev_f1 > dev_best_f1:

                    evaluate(model, config, dev_dataloader, id_to_vocabs, id_to_tags, logger, device, test=True)
                    dev_best_f1 = dev_f1
                    torch.save(model, os.path.join(config.save_dir, "medical_ner.ckpt"))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                msg = 'Iter: {} | Dev Loss: {:.4f} | Dev F1-micro: {:.4f} | {}'
                logger.info(msg.format(total_batch, dev_loss, dev_f1, improve))

            total_batch += 1

            if total_batch - last_improve > config.require_improve:
                """ 验证集f1超过500batch没上升，结束训练 """
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def main():

    device = 'cuda:0'
    #device = 'cpu'

    set_manual_seed(1234)
    print("设置随机数种子为20")

    train("train", device)

if __name__ == '__main__':

    print("Start")

    main()

    print("Successfully")

