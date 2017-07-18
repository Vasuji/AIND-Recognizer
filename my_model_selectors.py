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

    def __init__(self, all_word_sequences: dict,
                 all_word_Xlengths: dict,
                 this_word: str,
                 n_constant=3,
                 min_n_components=2,
                 max_n_components=10,
                 random_state=14, verbose=False):
        
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
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
            
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
            
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


    
#=============== BIC ========================



class SelectorBIC(ModelSelector):
    """ 
    Abbreviations:
        - BIC - Baysian Information Criterion
    
    Objective: 
        select the model with the lowest Bayesian Information Criterion(BIC) score
         
    About BIC:
        - SelectorBIC accepts argument of ModelSelector instance of base class
          with attributes such as: this_word, min_n_components, max_n_components
        - Maximises the likelihood of data whilst penalising large-size models
        - Used to scoring model topologies by balancing fit
          and complexity within the training set for each word
        - Avoids using CV by instead using a penalty term   
      
    References:
        [0] - http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        [1] - https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
        [2] - https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
        [3] - https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/8
        [4] - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf   
    """

        
    def bic_score(self, num_states):
        
        """
         BIC Equation:  BIC = -2 * log L + p * log N 
            L : is the likelihood of the fitted model
            p : is the number of parameters
            N : is the number of data points
                    
        Notes:
          -2 * log L    -> decreases with higher "p"
          p * log N     -> increases with higher "p"
          N > e^2 = 7.4 -> BIC applies larger "penalty term" in this case
            
        """
        
        model = self.base_model(num_states)
        log_likelihood = model.score(self.X, self.lengths)
        number_of_parameters = num_states ** 2 + 2 * num_states * model.n_features - 1
        score = -2 * log_likelihood + number_of_parameters * np.log(num_states)
        return score, model
        

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores
        
        """ Selection using BIC Model:
            - Lower the BIC score the "better" the model.
            - Loop from min_n_components to max_n_components
            - Find the lowest BIC score as the "better" model.
        """
      
        try:
            
            best_score = float("Inf") 
            best_model = None

            for num_states in range(self.min_n_components, self.max_n_components + 1):
                score, model = self.bic_score(num_states)
                if score < best_score:
                    best_score, best_model = score, model
            return best_model

        except:
            return self.base_model(self.n_constant)
    
  
    
    
#================= DIC ===========================    
    
    
    
    
    
class SelectorDIC(ModelSelector):
    
    """
      Abbreviations:
        - DIC - Discriminative Information Criterion
    
      Objective: 
       "select best model based on Discriminative Information Criterion
        Biem, Alain. A model selection criterion for classification: 
        Application to hmm topology optimization."
    
      About DIC:
        - In DIC we need to find the number of components where the difference is largest.
        The idea of DIC is that we are trying to find the model that gives a
        high likelihood (small negative number) to the original word and
        low likelihood (very big negative number) to the other words
        
        - In order to get an optimal model for any word, we need to run the model on all
        other words so that we can calculate the formula
        
        - DIC is a scoring model topology that scores the ability of a
        training set to discriminate one word against competing words.
        It provides a "penalty" if model likelihoods
        for non-matching words are too similar to model likelihoods for the
        correct word in the word set (rather than using a penalty term for
        complexity like in BIC)
        
        - Task-oriented model selection criterion adapts well to classification
        problems
        
        - Classification task accounts for Goal of model  (differs from BIC)
        
         
        References:
        [0] - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
         
    """
        
        
    def dic_score(self, num_states):
        """
          DIC Equation:
            DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
            i <- current_word
            DIC = log likelihood of the data belonging to model
                - avg of anti-log likelihood of data X and model M
                = log(P(original word)) - average(log(P(other words)))
               
            where anti-log likelihood means likelihood of data X and model M belonging
            to competing categories where log(P(X(i))) is the log-likelihood of the fitted
            model for the current word.
        
          Note:
            - log likelihood of the data belonging to model
            - anti_log_likelihood of data X vs model M

        """
        
        model = self.base_model(num_states)
        logs_likelihood = []
        
        for word, (X, lengths) in self.hwords.items():
            
            # likelihood of current word
            if word == self.this_word:
                current_word_likelihood = model.score(self.X, self.lengths)
                
            # if word != self.this_word:
            # likelihood of remaining words
            else:
                logs_likelihood.append(model.score(X, lengths))
             
        score = current_word_likelihood - np.mean(logs_likelihood)
        
        return score, model
    
    

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection based on DIC scores
        
        """Selection using DIC Model:
            - Higher the DIC score the "better" the model.
            - SelectorDIC accepts argument of ModelSelector instance of base class
              with attributes such as: this_word, min_n_components, max_n_components,
            - Loop from min_n_components to max_n_components
            - Find the highest BIC score as the "better" model.
        """

        
        try:
            best_score = float("-Inf")
            best_model = None
            
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, model = self.dic_score(num_states)
                if score > best_score:
                    best_score = score
                    best_model = model
            return best_model   

        except:
            return self.base_model(self.n_constant)

    
    
    
    
    
#====================== CV ===============================    
    
    
    
    
    
class SelectorCV(ModelSelector):
    """ 
        Abbreviations:
            - CV - Cross-Validation
            
            Objective: select best model based on average log Likelihood of cross-validation folds
        About CV:
            - Scoring the model simply using Log Likelihood calculated from
              feature sequences it trained on, we expect more complex models
              to have higher likelihoods, but doesn't inform us which would
              have a "better" likelihood score on unseen data. The model will
              likely overfit as complexity is added.
              
            - Estimate the "better" Topology model using only training data
              by comparing scores using Cross-Validation (CV).
              
            - CV technique includes breaking-down the training set into "folds",
              rotating which fold is "left out" of the training set.
              The fold that is "left out" is scored for validation.
              Use this as a proxy method of finding the
             "best" model to use on "unseen data".
             
              e.g. Given a set of word sequences broken-down into three folds
              using scikit-learn Kfold class object.
              
            - CV useful to limit over-validation
    
        References:
            [0] - http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
            [1] - https://www.r-bloggers.com/aic-bic-vs-crossvalidation/
            
    """
    
    def cv_score(self, num_states):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        
        CV Equation:
        
        
        """
        fold_scores = []
        split_method = KFold(n_splits = 3, shuffle = True, random_state = 1)
        
        
        for train_idx, test_idx in split_method.split(self.sequences):
            # Training sequences split using KFold are recombined
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)
            # Get test sequences
            test_X, test_length = combine_sequences(test_idx, self.sequences)
            # Record each model score
            model = self.base_model(num_states)
            fold_scores.append(model.score(test_X, test_length))
            
        # Compute mean of all fold scores
        score = np.mean(fold_scores)
            
        return score, model

    
    def select(self):
        
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        
        Selection using CV Model:
            - Higher the CV score the "better" the model.
            - Select "best" model based on average log Likelihood
              of cross-validation folds
            - Loop from min_n_components to max_n_components
            - Find the higher score(logL), the higher the better.
            - Score that is "best" for SelectorCV is the
              average Log Likelihood of Cross-Validation (CV) folds.
          
        """
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("Inf")
            best_model = None
            
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, model = self.cv_score(num_states)
                if score < best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)
        
        
        
    
    
    
    
    
    
    
    
    
    
    