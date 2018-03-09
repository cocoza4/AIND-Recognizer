import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.n_features = len(self.X[0])
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                parameters = n_components**2 + 2*n_components*self.n_features - 1
                BIC = -2 * logL + parameters * math.log(len(self.X))

                if BIC < best_score:
                    best_score = BIC
                    best_model = model

            except:
                if self.verbose:
                    print('Unable to perform model.score() {} hidden states'.format(n_components))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None
        M = len(self.hwords) - 1 # -1 because `self.this_word` is ignored
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                logL_others = 0
                for word in self.hwords:
                    if word != self.this_word:
                        X_other, lengths_other = self.hwords[word]
                        logL_others += model.score(X_other, lengths_other)

                DIC = logL - logL_others/(M-1)

                if DIC > best_score:
                    best_score = DIC
                    best_model = model

            except:
                if self.verbose:
                    print('Unable to perform model.score() {} hidden states'.format(n_components))

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_n_components = self.min_n_components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                logLs = []
                if len(self.lengths) == 1: # can't do cross validation, only 1 sample of word exists
                    model = self.base_model(n_components)
                    logLs = [model.score(self.X, self.lengths)]
                else:
                    kfold = KFold(n_splits=min(3, len(self.lengths)))

                    logLs = []
                    for train_idx, test_idx in kfold.split(self.sequences):
                        X_train, train_lengths = combine_sequences(train_idx, self.sequences)
                        X_test, test_lengths = combine_sequences(test_idx, self.sequences)
                        model = GaussianHMM(n_components=n_components, n_iter=1000, verbose=False).fit(X_train, train_lengths)

                        score = model.score(X_test, test_lengths)
                        logLs.append(score)
            except:
                if self.verbose:
                    print('Unable to train/score the model with {} hidden states'.format(n_components))
            finally:
                if logLs: # only check if model is trainable i.e. the `logLs` is not empty.
                    mean_logLs = np.mean(logLs)
                    if best_score < mean_logLs:
                        best_score = mean_logLs
                        best_n_components = n_components

        best_model = self.base_model(best_n_components) # train again with the whole X
        return best_model
