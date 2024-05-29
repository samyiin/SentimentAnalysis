"""
Sam's note
In the online dataset, it will compute the embedding of the sentence in runtime. I don't quite understand this design,
why not compute the embedding ahead of time?

Explanation of the design:
Each DataManger have a field: torch_iterators
torch_iterators is the following format: {"train": DataLoder, "test": DataLoder, "val": DataLoder}
DataLoder (torch iterator) takes a Dataset as input.
OnlineDataset is a child class of Dataset, we just rewrite the __getitem__ method.It also takes other parameters such as
 batch_size and shuffle, but that's not important for the explanation.
Pytorch Dataset does care your input, it only cares that the output is in "X, y" form (__getitem__). So Theoretically,
we can have a list of (embedding, sentiment value), and "cast" them into dataset object (so that we can construct
dataloader of pytorch).
What happened here, is we take a Sentence object. This object contains the text and sentiment value. Then OnlineDataset
just convert the text into embedding (by the callback function passed to it).
DataManager is just taking the data from SentimentTreeBank, and make pytorch DataLoder (and Dataset) instance out of it.

So technically we don't need a dataManager, we can just let SentimentTreeBank returns pytorch DataLoader instances. The
advantage here is that maybe if we are trying to use other framework, then we can just wrap dataloader to output
"dataloader" of the new framework.

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
        # the sentence interface: take sentence as a list of words(string)
        sent_emb = self.sent_func(sent.text, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    Sam's note
    This class have a simple usage: provide interface to get torch iterator for train, val and test set.
    One more thing it does: train set can be in phrase or sentence.
    """

    def __init__(self, use_sub_phrases, sentiment_dataset, sent_func, sent_func_kwargs, batch_size=50):
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
        # Here we use the "DataLoder" class of Torch:
        # train needs to shuffle, test need to be all in one batch: so we get prediction all at once.
        self.train_iterator = DataLoader(self.torch_datasets[TRAIN], batch_size=batch_size, shuffle= True)
        self.val_iterator = DataLoader(self.torch_datasets[VAL], batch_size=batch_size)
        self.test_iterator = DataLoader(self.torch_datasets[TEST], batch_size=batch_size)
        self.torch_iterators = {TRAIN: self.train_iterator,
                                VAL: self.val_iterator,
                                TEST: self.test_iterator}

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
