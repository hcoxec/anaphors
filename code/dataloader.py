from nltk import CFG
#from nltk import grammar
from nltk.parse.generate import generate
from pcfg import PCFG

import _jsonnet
import json

import random
from os.path import exists

from torch.utils.data import DataLoader, Dataset, TensorDataset
from ScaledDataset import ScaledDataset
import torch

from itertools import combinations

class objectview(object):
    '''
    An object that makes a dictionary's keys attributes of the object, so they can
    be called by subscripting (mimics the functionality of argparse)
    '''
    def __init__(self, d):
        self.__dict__ = d


class DataHandler(object):
    '''
    An object that handles loading of data, then divides it into a variety
    of structures suitable for training, eval, and analysis
    '''

    def __init__(self, args):
        ''''
        Takes an subscriptable arg dict as input, creates object attributes
        for relevant hyperparameters. Then runs preprocessing and data loading
        or generation automatically during init.
        '''
        self.opts = args
        lang_params = self.opts.lang_params  # determines the number of nouns and verbs
        self.n_count, self.v_count = lang_params[0], lang_params[1] #the number of nouns/verbs we want to initially generate
        self.noun_cap = lang_params[2] #the number of nouns we want to use in our game
        self.verb_cap = lang_params[3] #the number of verbs we want to use in our game
        self.gram_fn = self.opts.gram_fn

        # get dataset family
        self.dataset = self.opts.dataset

        # check which dataset we want
        self.setting = self.opts.setting

        # if we want to play around with uniform or powerlaw distributions
        self.probs = self.opts.probs

        # pretend data is larger for reinforce
        self.scaling_factor = int(self.opts.data_scaling)

        if self.probs == "mixed":
            self.init_grammar()
        else:
            self.grammar = self.init_grammar()

        if self.probs == "uniform" or self.probs == "mixed":
            self.initial_phrases = self.get_phrases()

        # whether to sample redundant sentences (True) or not (False)
        self.red = self.opts.red

        # sample from all generated phrases if necessary
        self.phrases = self.pick_phrases(self.red)

        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "EOS"}
        self.n_words = 1  # Count EOS
        self.build_dictionaries()

        # for game framework
        self.n_attributes = len(self.phrases[0].split())
        self.n_values = self.n_words

        self.idx_phrases = self.build_idx_phrases()
        self.hot_grid = self.build_hot_grid()

        #self.lang_type = self.opts.lang_type

        if self.check_for_data():
            self.load_dataset()
        else:
            #self.generate_compositional(self.idx_phrases, self.lang_type)
            self.build_dataset()
        
        # if self.comp_check_for_data():
        #     self.comp_load_dataset()
        # else:
        #     self.generate_compositional(self.idx_phrases)

    def check_for_data(self):
        '''
        Looks for a data directory rather than generating
        data from scratch
        '''
        data_paths = [f'data/{self.dataset}/{self.setting}_train_set.tar', f'data/{self.dataset}/{self.setting}_val_set.tar', f'data/{self.dataset}/{self.setting}_test_set.tar']
        does_exist = sum([exists(file) for file in data_paths]) == 3

        return does_exist

    def build_dataset(self):
        ''''
        Generates training test and val splits, then saves the splits
        to the data directory
        '''
        #self.train_set, self.val_set, self.test_set, self.comp_train_set, self.comp_val_set, self.comp_test_set = self.generate_splits(self.hot_grid, train=0.7, test=0.2)
        self.train_set, self.val_set, self.test_set = self.generate_splits(self.hot_grid, train=0.7, test=0.2)
        self.train_loader = self.get_loader(self.train_set, bs=self.opts.batch_size)
        self.val_loader = self.get_loader(self.val_set, bs=int(self.opts.batch_size/5))
        self.test_loader = self.get_loader(self.test_set, bs=int(self.opts.batch_size/5))

        torch.save(self.train_set, f'data/{self.dataset}/{self.setting}_train_set.tar')
        torch.save(self.val_set, f'data/{self.dataset}/{self.setting}_val_set.tar')
        torch.save(self.test_set, f'data/{self.dataset}/{self.setting}_test_set.tar')

        # self.train_comp_loader = self.get_loader(self.comp_train_set)
        # self.val_comp_loader = self.get_loader(self.comp_val_set)
        # self.test_comp_loader = self.get_loader(self.comp_test_set)

        # torch.save(self.comp_train_set, f'data/{self.setting}_comp_train_set.tar')
        # torch.save(self.comp_val_set, f'data/{self.setting}_comp_val_set.tar')
        # torch.save(self.comp_test_set, f'data/{self.setting}_comp_test_set.tar')

        #saving the corresponding phrase data and grammar
        with open(f"utils/dicts/{self.gram_fn}_dict.json", 'w+') as outfile:
            json.dump(self.index2word, outfile, indent=5)

        phrase_data = [self.phrase_train, self.phrase_val, self.phrase_test]
        names = ['train', 'val', 'test']
        for x in list(zip(phrase_data, names)):
            with open(f'data/{self.setting}_{x[1]}_phrase_set.txt', 'w+') as f:
                for phrase in x[0]:
                    f.write(f"{phrase}\n")

    def load_dataset(self):
        '''
        Loads a serialized dataset from local data directoy,
        and places it on the CPU
        '''

        self.train_set = torch.load(f'data/{self.dataset}/{self.setting}_train_set.tar', map_location=torch.device('cpu'))
        self.val_set = torch.load(f'data/{self.dataset}/{self.setting}_val_set.tar', map_location=torch.device('cpu'))
        self.test_set = torch.load(f'data/{self.dataset}/{self.setting}_test_set.tar', map_location=torch.device('cpu'))

        self.raw_train = self.train_set.tensors[0]
        self.raw_val = self.val_set.tensors[0]
        self.raw_test = self.test_set.tensors[0]

        self.train_loader = self.get_loader(self.train_set, bs=self.opts.batch_size)
        self.val_loader = self.get_loader(self.val_set, bs=int(self.opts.batch_size/5))
        self.test_loader = self.get_loader(self.test_set, bs=int(self.opts.batch_size/5))

        # self.comp_train_set = torch.load(f'datafinal/{self.setting}_comp_train_set.tar', map_location=torch.device('cpu'))
        # self.comp_val_set = torch.load(f'datafinal/{self.setting}_comp_val_set.tar', map_location=torch.device('cpu'))
        # self.comp_test_set = torch.load(f'datafinal/{self.setting}_comp_test_set.tar', map_location=torch.device('cpu'))

        # self.train_comp_loader = self.get_loader(self.comp_train_set)
        # self.val_comp_loader = self.get_loader(self.comp_val_set)
        # self.test_comp_loader = self.get_loader(self.comp_test_set)

    def pick_phrases(self, redundant=True):
        '''
        randomly sample from a list of generated phrases without replacement
        if redundant=False, then ignore all sentences with repeated nouns/verbs
        '''
        if self.probs == "uniform":
            red_phrases = []
            nonred_phrases = []
            for x in self.initial_phrases:
                y = x.split()
                if y[0] == y[3] or y[1] == y[4]:
                    red_phrases.append(x)
                else:
                    nonred_phrases.append(x)
            initial_phrases = red_phrases + random.sample([p for p in nonred_phrases], 20000-len(red_phrases)) #choose all possible generated redundant phrases, and sample from non-redundant for remaining data
            return self.initial_phrases

        elif self.probs == "powerlaw":
            initial_phrases = []
            for x in self.grammar.generate(100000):
                initial_phrases.append(x)
            if not redundant:
                filtered_phrases = []
                for x in initial_phrases:
                    y = x.split()
                    if y[0] != y[3] and y[1] != y[4]:
                        filtered_phrases.append(x)
                new_phrases = random.sample(filtered_phrases, 20000)
            else:
                new_phrases = random.sample(initial_phrases, 20000)
            return new_phrases

        elif self.probs == "mixed":
            if redundant:
                red_phrases = []
                plaw_phrases = []
                for x in self.initial_phrases:
                    y = x.split()
                    if y[0] == y[3] or y[1] == y[4]:
                        red_phrases.append(x)

                for x in self.plawgram.generate(100000):
                    plaw_phrases.append(x)
                filtered_plaw_phrases = []
                for x in plaw_phrases:
                    y = x.split()
                    if y[0] != y[3] and y[1] != y[4]:
                        filtered_plaw_phrases.append(x)
                new_plaw_phrases = random.sample(filtered_plaw_phrases, 20000-len(red_phrases))
                final_phrases = random.sample(new_plaw_phrases+red_phrases, 20000)
            return final_phrases

    def addSentence(self, sentence):
        '''
        preprocessing of data sentences, adding one word at a time to the language
        dictionaries
        '''
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        '''
        Adds a word to dictionaries that map from an index to a word and back,
        also tracks the number of words in the language
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def init_grammar(self):
        '''
        Loads grammar rules and terminals from file

        returns: nltk.CFG
        '''
        if self.probs == "uniform":
            with open("code/utils/words/verbs-discourse.txt") as verbfile:
                VERBS = [line.strip() for line in verbfile][:self.verb_cap]
            with open("code/utils/words/nouns-discourse.txt") as nounfile:
                NOUNS = [line.strip() for line in nounfile][:self.noun_cap]
            with open("code/utils/words/rules-discourse.txt") as rulefile:
                RULES = [line.strip() for line in rulefile]

            self.verbs = VERBS
            self.nouns = NOUNS

            v_rules = ['V -> \'' + this_word + '\'' for this_word in VERBS]
            n_rules = ['N -> \'' + this_word + '\'' for this_word in NOUNS]

            with open('code/utils/words/grammar-discourse.txt', 'w') as f:
                for item in RULES + v_rules + n_rules:
                    f.write("%s\n" % item)
            with open("code/utils/words/grammar-discourse.txt") as grammarfile:
                grammar = [line.strip() for line in grammarfile]
            return CFG.fromstring(grammar)

        elif self.probs == "powerlaw":
            with open("code/utils/words/plaw_grammar-discourse.txt") as grammarfile:
                grammar = [line.strip() for line in grammarfile]
            return PCFG.fromstring(grammar)

        elif self.probs == "mixed":
            with open("code/utils/words/verbs-discourse.txt") as verbfile:
                VERBS = [line.strip() for line in verbfile][:self.verb_cap]
            with open("code/utils/words/nouns-discourse.txt") as nounfile:
                NOUNS = [line.strip() for line in nounfile][:self.noun_cap]
            with open("code/utils/words/rules-discourse.txt") as rulefile:
                RULES = [line.strip() for line in rulefile]

            self.verbs = VERBS
            self.nouns = NOUNS

            v_rules = ['V -> \'' + this_word + '\'' for this_word in VERBS]
            n_rules = ['N -> \'' + this_word + '\'' for this_word in NOUNS]

            with open('code/utils/words/grammar-discourse.txt', 'w') as f:
                for item in RULES + v_rules + n_rules:
                    f.write("%s\n" % item)
            with open("code/utils/words/grammar-discourse.txt") as grammarfile:
                grammar = [line.strip() for line in grammarfile]
            self.unifgram = CFG.fromstring(grammar)

            with open("code/utils/words/plaw_grammar-discourse.txt") as grammarfile:
                grammar = [line.strip() for line in grammarfile]
            self.plawgram = PCFG.fromstring(grammar)

    def get_phrases(self):
        '''
        Returns all strings generated by the grammar
            input: None
            output: all_phrases [list]

        '''
        if self.probs == "uniform":
            return [' '.join(x) for x in generate(self.grammar)]
        elif self.probs == "mixed":
            return [' '.join(x) for x in generate(self.unifgram)]

    def build_dictionaries(self):
        '''
        Preprocessing helper, adds one sentence at a time from phrases to dictionaries
        !Requires there be a self.phrases attribute!
        '''
        for i in self.phrases:
            self.addSentence(i)

    def build_hot_grid(self):
        '''
        Builds one massive tensor with all data one-hot encoded
        !Requires there be a self.phrases attribute!

            input: None
            output: torch.tensor dims n_phrases X n_words
        '''
        master_grid = []
        for i, phrase in enumerate(self.phrases):
            this_grid = torch.zeros(len(phrase.split()), self.n_words)
            for w, word in enumerate(phrase.split()):
                word_idx = self.word2index[word]
                this_grid[w][word_idx] = 1
            master_grid.append(this_grid.view(-1))
        return torch.stack(master_grid)

    def build_idx_phrases(self):
        '''
        Creates an indexed representation of the data, with an index for each
        value, for each attribute
        !Requires there be a self.phrases attribute!

            input: None
            output: torch.tensor dims n_examples X n_attributes
        '''
        idx_phrases = []
        for i, phrase in enumerate(self.phrases):
            this_grid = torch.zeros(len(phrase.split()))
            for w, word in enumerate(phrase.split()):
                word_idx = self.word2index[word]
                this_grid[w] = word_idx
            idx_phrases.append(this_grid)
        return torch.stack(idx_phrases)

    def generate_splits(self, raw_data, train=0.75, val=0.1, test=0.15):
        '''
        Generates a random split of the data according to percentages, and returns
        a dataset object of the split. Assumes input and output are the same
        as in a reconstruction game.

        input:
            raw_data: training data (torch.tensor or list)
            train, val, test: percentages of data for each split, can be un-normalized
        '''
        #self.generate_compositional(self.idx_phrases, self.lang_type)

        idx = torch.randperm(raw_data.shape[0])
        data = raw_data[idx].view(raw_data.size())
        #comp_data = self.comp_signals[idx].view(self.comp_signals.size())

        total_prob = train + val + test
        train, val, test = train / total_prob, val / total_prob, test / total_prob
        train_c, val_c, test_c = int(len(data) * train), int(len(data) * val), int(len(data) * test)

        self.raw_train = data[:train_c]
        self.raw_val = data[train_c:-test_c]
        self.raw_test = data[-test_c:]

        # self.comp_train = comp_data[:train_c]
        # self.comp_val = comp_data[train_c:-test_c]
        # self.comp_test = comp_data[-test_c:]

        self.phrase_train = []
        self.phrase_val = []
        self.phrase_test = []

        for element in self.raw_train.view(len(self.raw_train), self.n_attributes, self.n_values).argmax(dim=2):
            phrase = ' '.join([self.index2word[e.item()] for e in element])
            self.phrase_train.append(phrase)

        for element in self.raw_val.view(len(self.raw_val), self.n_attributes, self.n_values).argmax(dim=2):
            phrase = ' '.join([self.index2word[e.item()] for e in element])
            self.phrase_val.append(phrase)

        for element in self.raw_test.view(len(self.raw_test), self.n_attributes, self.n_values).argmax(dim=2):
            phrase = ' '.join([self.index2word[e.item()] for e in element])
            self.phrase_test.append(phrase)

        train_set = ScaledDataset(self.raw_train, self.raw_train, scaling_factor=self.scaling_factor)
        val_set = TensorDataset(self.raw_val, self.raw_val)
        test_set = TensorDataset(self.raw_test, self.raw_test)

        # comp_train_set = TensorDataset(self.comp_train, self.raw_train)
        # comp_val_set = TensorDataset(self.comp_val, self.raw_val)
        # comp_test_set = TensorDataset(self.comp_test, self.raw_test)

        return train_set, val_set, test_set
        #return train_set, val_set, test_set, comp_train_set, comp_val_set, comp_test_set

    def get_loader(self, data_set, shuffle=True, bs=32, drop=True):
        '''
        Creates a torch dataloader of a dataset for use during training.
        input:
            data_set: usually a TensorDataset, but anything iterable works
            shuffle: randomly order batches each epoch of training
            bs: batch sizer for loader
        '''
        return DataLoader(data_set, batch_size=bs, shuffle=shuffle, drop_last=drop)

    def generate_compositional(self, semantics, ltype, signal_len=10, n_chars=26):
        '''
        For creating the handcrafted languages for the first experiment with the Receiver only

        Generates a perfectly compositional language for a given semantic
        representation.

        input:
            semantics: list of index semantics generated by build_idx_phrases or similar
            signal_len: number of chars in each signal
            n_chars: number of chars to choose from in the language

            !signal_len must be divisible by n_attributes!
        '''
        self.idx2signal = {}
        subsignals = [x for x in combinations(range(2, n_chars), signal_len // 5)]
        random.shuffle(subsignals)
        for i, idx in enumerate(self.index2word.keys()):
            self.idx2signal[idx] = subsignals[i]

        #the "Pronoun" language: adds 'did too'/'pronoun' signal character for repeated noun/verb
        if ltype == "tok":
            self.idx2signal[len(self.idx2signal)] = 26 #pronoun token
            self.idx2signal[len(self.idx2signal)] = 27 #'did too' token
        all_signals = []

        #the "No Elision" language: repeats the original signal for a repeated noun/verb
        if ltype == "comp":
            for sem in semantics:
                one_signal = []
                for pos in sem: # not redundant
                    one_signal.extend(self.idx2signal[int(pos)])
                one_signal.append(0)
                all_signals.append(one_signal)

        #the "Pro-drop" language: removes/doesn't append any signal character for a repeated noun/verb
        elif ltype == "null":
            for sem in semantics:
                one_signal = []
                if sem[0] == sem[3] and sem[1] == sem[4]: #fully redundant
                    for i, pos in enumerate(sem):
                        if i != 3 and i != 4:
                            one_signal.extend(self.idx2signal[int(pos)])
                elif sem[0] == sem[3]: #partially redundant
                    for i, pos in enumerate(sem):
                        if i != 3:
                            one_signal.extend(self.idx2signal[int(pos)])
                elif sem[1] == sem[4]: #partially redundant
                    for i, pos in enumerate(sem):
                        if i != 4:
                            one_signal.extend(self.idx2signal[int(pos)])
                else:
                    for pos in sem: #not redundant
                        one_signal.extend(self.idx2signal[int(pos)])
                while len(one_signal) != 11:
                    one_signal.append(0) #for EOS
                all_signals.append(one_signal)


        elif ltype == "tok":
            for sem in semantics:
                one_signal = []
                if sem[0] == sem[3] and sem[1] == sem[4]: #fully redundant
                    for i, pos in enumerate(sem):
                        if i == 3:
                            one_signal.append(self.idx2signal[len(self.idx2signal) - 2])
                        if i == 4:
                            one_signal.append(self.idx2signal[len(self.idx2signal) - 1])
                        else:
                            one_signal.extend(self.idx2signal[int(pos)])
                elif sem[0] == sem[3]: #partially redundant
                    for i, pos in enumerate(sem):
                        if i == 3:
                            one_signal.append(self.idx2signal[len(self.idx2signal) - 2])
                        else:
                            one_signal.extend(self.idx2signal[int(pos)])
                elif sem[1] == sem[4]: #partially redundant
                    for i, pos in enumerate(sem):
                        if i == 4:
                            one_signal.append(self.idx2signal[len(self.idx2signal) - 1])
                        else:
                            one_signal.extend(self.idx2signal[int(pos)])
                else:
                    for i, pos in enumerate(sem): #not redundant
                        one_signal.extend(self.idx2signal[int(pos)])
                while len(one_signal) != 11:
                    one_signal.append(0) #for EOS
                all_signals.append(one_signal)

        if ltype == "tok":
            target_signals = self.build_any_grid(all_signals, 28)
        else:
            target_signals = self.build_any_grid(all_signals, 26)

        self.comp_signals = torch.tensor(all_signals)
        self.comp_data = target_signals

    def build_any_grid(self, idx_signals, n_chars):
        '''
        Generalization of build_idx_phrases, creates an index-based semantic
        representation not strictly tied to self.phrases. Intended to create
        a one hot grid for compositional signals, to train one agent on signal
        meaning pairs

        input:
            idx_signals: list n_examples X n_attributes
            n_chars: number of chars in language
        '''
        master_grid = []
        longest = 0
        for phrase in idx_signals:
            if len(phrase) > longest:
                longest = len(phrase)
        for i, phrase in enumerate(idx_signals):
            #this_grid = torch.zeros(len(idx_signals[0]), n_chars)
            this_grid = torch.zeros(longest, n_chars)
            for w, word in enumerate(phrase):
                this_grid[w][word] = 1
            master_grid.append(this_grid)
        return torch.stack(master_grid)

    def __len__(self):
        return len(self.phrases)

def main():
    args = objectview(json.loads(_jsonnet.evaluate_file('config.jsonnet')))
    data = DataHandler(args)
    # print(data.index2word)
    # print(data.idx_phrases[0])

if __name__ == "__main__":
    main()