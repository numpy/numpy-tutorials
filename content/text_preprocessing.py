import pandas as pd
import argparse
import numpy as np
import re # (https://docs.python.org/3/library/re.html) for tokenising textual data 
import string # (https://docs.python.org/3/library/string.html) for string operations  

class TextPreprocess:
    """Text Preprocessing for a Natural Language Processing model."""


    def cleantext(self, df, text_column, remove_stopwords = True, remove_punc = True):
        """Function to clean text data by removing stopwords, tags and punctuation.

        Parameters
        ----------
        df : pandas dataframe 
            The dataframe housing the input data.
        text_column : str
            Column in dataframe whose text is to be cleaned.
        remove_stopwords : bool
            if True, remove stopwords from text
        remove_punc : bool
            if True, remove punctuation suymbols from text

        Returns
        -------
        Numpy array 
            Cleaned text.

        """
        data = df
        # converting all characters to lowercase 
        data[text_column] = data[text_column].str.lower()

        # List of common stopwords taken from https://gist.github.com/sebleier/554280
        stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
            "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
            "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
            "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
            "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
            "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
            "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
            "your", "yours", "yourself", "yourselves" ]

        def remove_stopwords(data, column):
            data[f'{column} without stopwords'] = data[column].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
            return data

        def remove_tags(string):
            result = re.sub('<*>','',string)
            return result

        # remove html tags and brackets from text 
        if remove_stopwords:
            data_without_stopwords = remove_stopwords(data, text_column)
            data_without_stopwords[f'clean_{text_column}']= data_without_stopwords[f'{text_column} without stopwords'].apply(lambda cw : remove_tags(cw))
        if remove_punc:
            data_without_stopwords[f'clean_{text_column}'] = data_without_stopwords[f'clean_{text_column}'].str.replace('[{}]'.format(string.punctuation), ' ', regex = True)

        X = data_without_stopwords[f'clean_{text_column}'].to_numpy()

        return X 

    def split_data (self, X, y, split_percentile):
        """Function to split data into training and testing data.

        Parameters
        ----------
        X : Numpy Array
            Contains textual data.
        y : Numpy Array
            Contains target data.
        split_percentile : int
            Proportion of training to testing data.
         

        Returns
        -------
        Tuple 
            Contains numpy arrays of test and training data.

        """
        y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
        arr_rand = np.random.rand(X.shape[0])
        split = arr_rand < np.percentile(arr_rand, split_percentile)
        X_train = X[split]
        y_train = y[split]
        X_test =  X[~split]
        y_test = y[~split]

        return (X_train, y_train, X_test, y_test)
   
    
    def sent_tokeniser (self, x):
        """Function to split text into sentences.

        Parameters
        ----------
        x : str
            piece of text

        Returns
        -------
        list 
            sentences with punctuation removed.

        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', x)
        sentences.pop()
        sentences_cleaned = [re.sub(r'[^\w\s]', '', x) for x in sentences]
        return sentences_cleaned
    
    def word_tokeniser(self, text):
        """Function to split text into tokens.

        Parameters
        ----------
        x : str
            piece of text

        Returns
        -------
        list 
            words with punctuation removed.

        """
        tokens = re.split(r"([-\s.,;!?])+", text)
        words = [x for x in tokens if (x not in '- \t\n.,;!?\\' and '\\' not in x)]
        return words

    def loadGloveModel(self, emb_path):
        """Function to read from the word embedding file.

        Returns
        -------
        Dict 
            mapping from word to corresponding word embedding.

        """
        print("Loading Glove Model")
        File = emb_path
        f = open(File,'r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel)," words loaded!")
        return gloveModel

        
    def text_to_paras(self, text, para_len): 
        """Function to split text into paragraphs.

        Parameters
        ----------
        text : str
            piece of text
            
        para_len : int
            length of each paragraph

        Returns
        -------
        list 
            paragraphs of specified length.

        """
        # split the speech into a list of words 
        words = text.split()
        # obtain the total number of paragraphs
        no_paras = int(np.ceil(len(words)/para_len))
        # split the speech into a list of sentences 
        sentences = self.sent_tokeniser(text)
        # aggregate the sentences into paragraphs
        k, m = divmod(len(sentences), no_paras)
        agg_sentences = [sentences[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(no_paras)]
        paras = np.array([' '.join(sents) for sents in agg_sentences])
        
        return paras 
        
