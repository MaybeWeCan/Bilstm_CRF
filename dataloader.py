from torch.utils.data import Dataset
import torch


class RMRB_Dataset(Dataset):
    '''
        读取数据进来，包装成：seq, seq格式
    '''
    def __init__(self, file_name, word_to_id, tag_to_id):
        self.datas = []

        with open(file_name, "r", encoding='utf-8') as f:
            words, tags = [], []
            seq_count = 0

            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) >= 5:
                        seq_count += 1

                        #if 1 not in tags:
                        #    words, tags = [], []
                        #    continue

                        self.datas.append((words,tags))
                        words, tags = [], []
                    else:
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]

                    # "UNK"
                    words_id = word_to_id.get(word, 1)
                    tags_id = tag_to_id.get(tag, 0)
                    words.append(words_id)
                    tags.append(tags_id)

                    # words.append(word)
                    # tags.append(tag)



        self.datas = sorted(self.datas, key=lambda x: len(x[0]))

    def __getitem__(self, item):
        return self.datas[item]

    def __len__(self):
        return len(self.datas)


def RMRB_collate_fn(batch):

    '''PAD and MASK'''
    max_length = max([len(sentence[0]) for sentence in batch])

    batch_seqs = []
    batch_tags = []
    batch_mask = []

    # PAD and MASK
    for char_seq,tag in batch:

        # 注意： vocab中0表示PAD, tag中0表示 ‘O’
        padding = [0] * (max_length - len(char_seq))

        batch_seqs.append(char_seq + padding)
        batch_tags.append(tag + padding)
        batch_mask.append([1] * len(char_seq) + padding)

    # Tensor
    batch_seqs = torch.LongTensor(batch_seqs)
    batch_tags = torch.LongTensor(batch_tags)

    # 必须这样，不然crf会报错：all only supports torch.uint8 and torch.bool dtypes
    batch_mask = torch.tensor(batch_mask, dtype=torch.uint8)

    return batch_seqs, batch_tags, batch_mask










