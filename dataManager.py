"""
Sam's note
In the online dataset, it will compute the embedding of the sentence in runtime. I don't quite understand this design,
why not compute the embedding ahead of time?
Explanation of the design:
Each DataManger have a field: torch_iterators
torch_iterators is the following format: {"train": DataLoder, "test": DataLoder, "val": DataLoder}
DataLoder takes a Dataset as input. OnlineDataset is a child class of Dataset, we just rewrite the __getitem__ method.
It also takes other parameters such as batch_size and shuffle, but that's not important for the explanation.

So when we train, we will just pass the torch_iterator["train"] to get the pyTorch DataLoder.
Same for getting the validation and test data.
"""
import numpy as np
from torch.utils.data import DataLoader, Dataset

TRAIN = "train"
VAL = "val"
TEST = "test"


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    This is a Type of Torch Dataset type, the difference when loading the data, it will embed the data into vectors:
    Sentence object -> numpy 1-dim vector.
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        Sam's Note:
        in this context, sentences is a list of Sentence object, sent_func is a embedding function that will embed a
        sentence into vectors, (so we can pass into neuronetwork.)
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, use_sub_phrases, sentiment_dataset, sent_func, sent_func_kwargs, batch_size=50 ):
        """
        :param use_sub_phrases: decide if we want to train with sub phrases
        :param batch_size: the batch size in training
        :param sentiment_dataset: It should be an object of dataLoder.SentimentTreeBank
        :param sent_func: this function will be passed to OnlineDataset
        :param sent_func_kwargs: this is parameters of the above function
        """

        # load the dataset: the type is list of Sentence object.
        self.sentiment_dataset = sentiment_dataset
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # record the embedding function and the embedding keywords
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

        # map data splits to torch datasets and iterators
        # this will be in format of {"train": OnlineDataset, "test": OnlineDataset, "val: : OnlineDataset}
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        # Here we use the "DataLoder" class of Torch
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape
