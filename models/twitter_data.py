import os
from collections import defaultdict, Counter
import numpy as np
import random
import nltk
from tqdm import tqdm
from models.data_utils import Sample, Batch

class TwitterData:
    def __init__(self, args):
        self.args = args

        #note: use 20k most frequent words
        self.UNK_WORD = '<unk>'
        self.PAD_WORD = '<pad>'

        # list of batches
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []

        self.word2id = {}
        self.id2word = {}

        self.train_samples = None
        self.valid_samples = None
        self.test_samples = None

        self.train_samples, self.valid_samples, self.test_samples = self._create_data()

        self.preTrainedEmbedding = None
        # [num_batch, batch_size, maxStep]
        self.train_batches = self._create_batch(self.train_samples)
        self.val_batches = self._create_batch(self.valid_samples)

        # note: test_batches is none here
        self.test_batches = self._create_batch(self.test_samples)


    def getVocabularySize(self):
        assert len(self.word2id) == len(self.id2word)
        return len(self.word2id)

    def _create_batch(self, all_samples, tag='test'):
        all_batches = []
        if tag == 'train':
            random.shuffle(all_samples)
        if all_samples is None:
            return all_batches

        num_batch = len(all_samples)//self.args.batchSize + 1
        for i in range(num_batch):
            samples = all_samples[i*self.args.batchSize:(i+1)*self.args.batchSize]

            if len(samples) == 0:
                continue

            batch = Batch(samples)
            all_batches.append(batch)

        return all_batches

    def _create_samples(self, file_path, test=False):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            all_samples = []
            for idx, line in enumerate(tqdm(lines)):
                # if idx == 100000:
                #     break
                if not test:
                    words = line.split()
                    label = int(words[-1].strip())
                    words = words[0:-1]
                else:
                    label = 0
                    start_idx = line.find(',')
                    line = line[start_idx+1:]
                    words = line.split()
                # for sentence that has only one word, duplicate that word to avoid bugs in attention
                if len(words) == 1:
                    words.append(words[-1])
                word_ids = []

                words = words[:self.args.maxSteps]
                for word in words:
                    if word in self.word2id.keys():
                        id_ = self.word2id[word]
                    else:
                        id_ = self.word2id[self.UNK_WORD]
                    word_ids.append(id_)
                while len(word_ids) < self.args.maxSteps:
                    word_ids.append(self.word2id[self.PAD_WORD])
                while len(words) < self.args.maxSteps:
                    words.append(self.PAD_WORD)
                sample = Sample(data=word_ids, words=words,
                                steps=self.args.maxSteps, label=label, flag_word=self.word2id[self.PAD_WORD])
                all_samples.append(sample)

        return all_samples

    def _create_data(self):

        train_path = os.path.join(self.args.twitterDir, self.args.twitterTrainFile)
        val_path = os.path.join(self.args.twitterDir, self.args.twitterValFile)
        test_path = os.path.join(self.args.twitterDir, self.args.twitterTestFile)

        print('Building vocabularies for twitter dataset')
        self.word2id, self.id2word = self._build_vocab(train_path)

        print('Building training samples!')
        train_samples = self._create_samples(train_path)
        random.shuffle(train_samples)
        val_samples = self._create_samples(val_path)
        test_samples = self._create_samples(test_path, test=True)

        return train_samples, val_samples, test_samples

    def _read_sents(self, filename, test=False):
        with open(filename, 'r') as file:
            all_sents = []
            all_words = []
            lines = file.readlines()
            for idx, line in enumerate(tqdm(lines)):
                # if idx == 100000:
                #     break
                if not test:
                    words = line.split()[0:-1]
                    all_words.extend(words)
                # we do not have labels for test data
                else:
                    words = line.split()
                    all_words.extend(words)

                # make every sample have the same length
                if len(words) > self.args.maxSteps:
                    all_sents.append(words[:self.args.maxSteps])
                    continue

                while len(words) < self.args.maxSteps:
                    words.append(self.PAD_WORD)
                all_sents.append(words)

        return all_sents, all_words

    def _build_vocab(self, filename):
        all_sents, all_words = self._read_sents(filename)

        counter = Counter(all_words)

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        #print(count_pairs[300000])
        # keep the most frequent vocabSize words, including the special tokens
        # -1 means we have no limits on the number of words
        if self.args.vocabSize != -1:
            count_pairs = count_pairs[0:self.args.vocabSize-2]

        count_pairs.append((self.UNK_WORD, 100000))
        count_pairs.append((self.PAD_WORD, 100000))

        if self.args.vocabSize != -1:
            assert len(count_pairs) == self.args.vocabSize

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        id_to_word = {v: k for k, v in word_to_id.items()}

        return word_to_id, id_to_word

    def get_batches(self, tag='train'):
        if tag == 'train':
            return self._create_batch(self.train_samples, tag='train')
        elif tag == 'val':
            return self.val_batches
        else:
            return self.test_batches
