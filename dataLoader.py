import os
import random

POSITIVE_SENTIMENT = 1.
NEGATIVE_SENTIMENT = 0.
NEUTRAL_SENTIMENT = -1.

SENTIMENT_NAMES = {
    POSITIVE_SENTIMENT: "Positive",
    NEUTRAL_SENTIMENT: "Neutral",
    NEGATIVE_SENTIMENT: "Negative"
}
SENTS_PATH = "SOStr.txt"
TREES_PATH = "STree.txt"
DICT_PATH = "dictionary.txt"
LABELS_path = "sentiment_labels.txt"
'''
Sam's Note:
Each line of the dictionary.txt is a map between a phrase to it's phrase id, in format of "PHRASE | PHRASE_ID"
Each line of sentiment_labels.txt is mapping from PHRASE_ID to sentiment score. 
So together we have a map from a phrase to it's sentiment score. 

Each line of SOStr.txt and STree.txt is corresponding to each other. 
Each line of SOStr.txt is a sentence, with each word seperated by |. 
Each line of STree.txt is a list of numbers, seperated by |. If the sentence is length n, then the first n number in 
the number list corresponds to the n words. The meaning of the number is the "place" of the parent. So say the sentence 
is "hi|hello|world|.", it's length of four. Then there will be list of numbers of length 7: 6|6|5|5|7|7|0. that means 
"hi" and "hello" have number 6, their parent is what's in the 6th place. "world" and "." have number 5, their parent is
 what's in the 5th place. and finally, the "what's in the 5th place" and "what's in the 6th place" have number 7, so 
 their parent is number 7, which has a number of 0, and we stop when the number is 0. 
 Then we know that  "what's in the 5th place" is a phrase: "world .", and similar for "what's in the 6th place". 
 Finally the root, "what's in the 5th place", is the full sentence. 
 So it is basically a tree, the leaves are the words, nodes in the middle (might have more than 1 layer) are some 
 phrases, and then the root is the entire sentence. 

SentimentTreeBank._build_dataset() explanation:
Conceptually, the first step is generating a dictionary, mapping from a phrase to the sentiment score of the phrase. 
* this is done by loading the dictionary.txt and sentiment_labels.txt
Then, for each sentence, for each word and phrase in the sentence tree (explained above), we found the sentiment score 
of the word/phrase in the sentiment score dictionary above. 
'''


def get_sentiment_class_from_val(sentiment_val: float):
    if sentiment_val <= 0.4:
        return NEGATIVE_SENTIMENT
    elif sentiment_val >= 0.6:
        return POSITIVE_SENTIMENT
    else:
        return NEUTRAL_SENTIMENT


class SentimentTreeNode(object):
    """
    root node have field "text", and connects to child nodes, and parent nodes
    for example: root node contains text [hi, hello, world]
    then child node_1 contain text [hi]
    child node_2 contain text [hello, world]
    """

    def __init__(self, text: list, sentiment_val: float, min_token_idx: int, children=[], parent=None):
        self.text = text
        self.sentiment_val = sentiment_val
        self.min_token_idx = min_token_idx
        self.sentiment_class = get_sentiment_class_from_val(sentiment_val)
        self.children = children
        self.parent = parent


class Sentence(object):
    """
    Represents a sentence in sentiment tree bank.
    Sentence can be considered as "unpacking" of <class>SentimentTreeNode instance
    Sentence will/should be a root node, <class>SentimentTreeNode can be child node or root node
    You can access the sentence text by sent.text
    This will give you a list of tokens (strings) in the order that they appear in the sentence.
    sent.sentiment_class is the coding of the annotated sentiment polarity of the sentence.
    sent.sentiment_val is the exact annotated sentiment value in the range [0,1]
    """

    def __init__(self, sentence_root: SentimentTreeNode):
        self.root = sentence_root
        self.text = sentence_root.text
        self.sentiment_class = sentence_root.sentiment_class
        self.sentiment_val = sentence_root.sentiment_val

    def _get_leaves_recursively(self, cur_root: SentimentTreeNode):
        if len(cur_root.children) == 0:
            return [cur_root]
        else:
            cur_leaves = []
            for child in cur_root.children:
                # combine two lists: python syntax list + list
                cur_leaves += self._get_leaves_recursively(child)
            return cur_leaves

    def get_leaves(self):
        '''
        recursively scan all child nodes in type of <class>SentimentTreeNode
        :return: a list of all child nodes, in correct order: order of how the root is divided into child node
        '''
        return self._get_leaves_recursively(self.root)

    def __repr__(self):
        return " ".join(self.text) + " | " + SENTIMENT_NAMES[self.sentiment_class] + " | " + str(self.sentiment_val)


class SentimentTreeBank(object):
    """
    The main object that represents the stanfordSentimentTreeBank dataset. Can be used to access the
    examples and some other utilities.
    self.sentence: a list of <class>Sentence instances
    """

    def __init__(self, path="StanfordSentimentTreebank", split_ratios=(0.8, 0.1, 0.1), split_words=True):
        """

        :param path: relative or absolute path to the datset directory
        :param split_ratios: split ratios for train, validation and test. please do not change!
        :param split_words: whether to split tokens with "-" and "/" symbols. please do not change!
        """
        self._base_path = path
        self.split_words = split_words
        # here we load the SOStr.txt into list of lists, each sentence is a list, we have a list of sentences
        sentences = self._read_sentences()
        # here we take sentence tree (explained above), and get sentiment for each nodes on the tree,
        # return a list of Sentence objects. Sentence object have sentiment class and value for each node in it's tree.
        self.sentences = self._build_dataset(sentences)
        '''
        get rid of dash and slash in the words, and update the parents in the sentence tree. 
        '''
        if self.split_words:
            for sent in self.sentences:
                leaves = sent.get_leaves()
                # as explained above, the leaves are the words
                for node in leaves:
                    node_text = node.text
                    splitted = node_text[0].split("-")
                    splitted_final = []
                    for s in splitted:
                        splitted_final.extend(s.split("\\/"))
                    # the for loop up until this point get rid of all the - and / in the words
                    if len(splitted_final) > 1 and all([len(s) > 0 for s in splitted_final]):
                        leaves = [SentimentTreeNode([s], node.sentiment_val,
                                                    min_token_idx=node.min_token_idx, parent=node) for
                                  s in splitted_final]
                        node.text = splitted_final
                        node.children = leaves
                        cur_parent = node.parent
                        # after getting rid of the - and /, we need to update the parents too, because they are phrase
                        # made from the leaves (words). So when the word changes, the parents changes.
                        while cur_parent != None:
                            cur_parent.text = []
                            for child in cur_parent.children:
                                cur_parent.text.extend(child.text)
                            cur_parent = cur_parent.parent
                sent.text = sent.root.text

        assert len(split_ratios) == 3
        assert sum(split_ratios) == 1
        self.split_ratios = split_ratios

    def _read_sentences(self):
        '''

        :return: sentences: list of list. represent a paragraph, each sentence is a list of words
        '''
        sentences = []
        with open(os.path.join(self._base_path, SENTS_PATH), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for i, line in enumerate(lines):
                if len(line.strip()) == 0:
                    continue
                line_content = line.strip()
                tokens = line_content.split("|")
                tokens = [t.lower().replace("-lrb-", "(").replace("-rrb-", ")") for t in tokens]
                sentences.append(tokens)
        return sentences

    def _build_dataset(self, sentences):
        '''
        in order to create a Sentence instance, we need a fully grown SentimentTreeNode root instance
        so we build all child of the root, and use this root to create a Sentence object
        :param sentences: list of list, can be view as paragraph
        :return: labeled_sentences: a list of <class>Sentence
        '''
        phrases_dictionary = {}

        # extract phrases: see note for dictionary.txt
        with open(os.path.join(self._base_path, DICT_PATH), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]
            for line in lines:
                phrase, phrase_id = line.strip().split("|")
                # "-lrb-" = left round bracket, similar for rrb
                phrases_dictionary[phrase.lower().replace("-lrb-", "(").replace("-rrb-", ")")] = int(phrase_id)

        # extract labels: see note for sentiment_labels.txt
        with open(os.path.join(self._base_path, LABELS_path), "r", encoding="utf-8") as f:
            lines = [l.strip().split("|") for l in f.read().split("\n")[1:-1]]
            labels_dict = {int(l[0]): float(l[1]) for l in lines}

        # helper function
        def get_val_from_phrase(phrase_tokens_list):
            try:
                return labels_dict[phrases_dictionary[" ".join(phrase_tokens_list)]]
            except:
                print("couldn't find key!")

        # load the sentences tree structures
        tree_pointers = []
        with open(os.path.join(self._base_path, TREES_PATH), "r") as f:
            for line in f.readlines():
                sent_pointers = [int(p) for p in line.strip().split("|")]
                tree_pointers.append(sent_pointers)
        assert len(tree_pointers) == len(sentences)

        # create Sentence instances with tree of SentimentTreeNodes
        labeled_sentences = []
        for sent, sent_pointers in zip(sentences, tree_pointers):
            try:
                children_dict = {i: [] for i in range(len(sent_pointers))}
                # i is the index (of sentence), p is the pointer (the number in each row of STree.txt)
                for i, p in enumerate(sent_pointers):
                    if i < len(sent):
                        node_text = [sent[i]]
                        node = SentimentTreeNode(text=node_text, sentiment_val=get_val_from_phrase(node_text),
                                                 min_token_idx=i)
                    else:
                        children = children_dict[i]
                        children = sorted(children, key=lambda n: n.min_token_idx)
                        node_text = []
                        for child in children:
                            node_text.extend(child.text)
                        node = SentimentTreeNode(text=node_text, sentiment_val=get_val_from_phrase(node_text),
                                                 children=children, min_token_idx=children[0].min_token_idx)
                        for child in children:
                            child.parent = node
                    if p > 0:
                        children_dict[p - 1].append(node)
                    last_node = node
                new_sentence = Sentence(last_node)
                if new_sentence.sentiment_class == NEUTRAL_SENTIMENT:
                    continue
                labeled_sentences.append(new_sentence)
            except Exception as e:
                print("couldn't parse sentence!")
                print(sent)
                raise e
        random.Random(1).shuffle(labeled_sentences)  # shuffle but with the same shuffle each time
        return labeled_sentences

    def get_train_set(self):
        """
        :return: list of Sentence instances for the train part of the dataset
        """
        if not hasattr(self, "_train_set"):
            self._train_set = self.sentences[:int(self.split_ratios[0] * len(self.sentences))]
        return self._train_set

    def _extract_all_phrases(self, root):
        # This is not getting training set.
        # why do we exclude neutral sentiment?
        phrases = [Sentence(root)] if root.sentiment_class != NEUTRAL_SENTIMENT else []
        if len(root.text) == 1:
            return []
        for child in root.children:
            phrases += self._extract_all_phrases(child)
        # we are returning a lot of "root" object (sentimentTreeNode) that is cast into Sentence object.
        return phrases

    def get_train_set_phrases(self):
        """
        :return: list of Sentence instances for the train part of the dataset including all sub-phrases
        """
        # the "if not instance" is just to make sure it is not initialized before. Is this even a good design?
        if not hasattr(self, "_train_set_phrases"):
            train_set = self.get_train_set()
            train_set_phrases = []
            for sent in train_set:
                train_set_phrases += self._extract_all_phrases(sent.root)
            self._train_set_phrases = train_set_phrases
        return self._train_set_phrases  # a list of Sentence object (cast)

    def get_test_set(self):
        """
        :return: list of Sentence instances for the test part of the dataset
        """
        if not hasattr(self, "_test_set"):
            begin_index = int(self.split_ratios[0] * len(self.sentences))
            end_index = int(sum(self.split_ratios[:2]) * len(self.sentences))
            self._test_set = self.sentences[begin_index:end_index]
        return self._test_set

    def get_validation_set(self):
        """
        :return: list of Sentence instances for the validation part of the dataset
        """
        if not hasattr(self, "_validation_set"):
            self._validation_set = self.sentences[int(sum(self.split_ratios[:2]) * len(self.sentences)):]
        return self._validation_set

    def get_train_word_counts(self):
        """
        :return: dictionary of all words in the train set with their frequency in the train set.
        """
        if not hasattr(self, "_train_word_counts"):
            word_counts = {}
            for sent in self.get_train_set():
                for word_node in sent.get_leaves():
                    assert len(word_node.text) == 1
                    word_text = word_node.text[0]
                    if word_text in word_counts:
                        word_counts[word_text] += 1
                    else:
                        word_counts[word_text] = 1
            self._train_word_counts = word_counts

        return self._train_word_counts

    def get_word_counts(self):
        """
        :return: dictionary of all words in the dataset with their frequency in the whole dataset.
        """
        if not hasattr(self, "_word_counts"):
            word_counts = {}
            for sent in self.sentences:
                for word_node in sent.get_leaves():
                    assert len(word_node.text) == 1
                    word_text = word_node.text[0]
                    if word_text in word_counts:
                        word_counts[word_text] += 1
                    else:
                        word_counts[word_text] = 1
            self._word_counts = word_counts

        return self._word_counts



