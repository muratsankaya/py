import math
import re
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class Bayes_Classifier:
    """
    This is a Bayes Classifier that classifies movie ratings as 1(bad review) and 5(good review)
    """
    def __init__(self):
        self.ones = None
        self.fives = None

    def test_splitted_data(self, ones, fives):
        for review in ones:
            assert review[0] == "1"

        for review in fives:
            assert review[0] == "5"

        print("Test split is successful")

    def process_word(self, word):

        # Removing capitilization
        word = word.lower()

        # Removig stop words
        if word in set(stopwords.words('english')): 
            return None
        
        # Removing punctuation
        if word in string.punctuation:
            return None

        # Apply Porter stemming
        word = self.porter.stem(word)

        return word
        
    # def process_lines(self, lines, label):
    #     bigram_count = 0
    #     bigram_counts = {}
    #     for (rating, review) in lines:
    #         words = word_tokenize(review)
    #         processed_words = []

    #         for word in words:
    #             processed_word = self.process_word(word)
                
    #             if processed_word:
    #                 processed_words.append(processed_word)

    #         for bigram in bigrams(processed_words):
    #             bigram_count += 1
    #             bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    #     setattr(self, label, (bigram_counts, bigram_count))

    def process_lines(self, lines, label):
        word_count = 0
        word_counts = {}
        for (rating, review) in lines:
            words = word_tokenize(review)
            
            for word in words:
                processed_word = self.process_word(word)
                if processed_word:
                    word_count += 1
                    word_counts[processed_word] = word_counts.get(processed_word, 0) + 1


        setattr(self, label, (word_counts, word_count))

    def split_by_rating(self, ones, fives, lines):
        for line in lines:
            splitted_line = line.rstrip().split("|")
            ones.append((splitted_line[0], splitted_line[2])) \
                if splitted_line[0] == "1" else \
                    fives.append((splitted_line[0], splitted_line[2]))

    def train(self, lines):
        ones, fives = [], []
        
        self.split_by_rating(ones, fives, lines)

        # self.test_splitted_data(ones, fives)

        # Initialize porter stemmer to be used in word processing
        self.porter =  PorterStemmer()

        self.process_lines(ones, "ones")
        self.process_lines(fives, "fives")

        # print(self.fives)

    def classify(self, lines):
        predictions = []
        vocab_size = len(set(self.ones[0].keys()).union(set(self.fives[0].keys())))

        for line in lines:
            words = word_tokenize(line.split("|")[2])

            prob_of_one = math.log(self.ones[1] / (self.ones[1] + self.fives[1]))
            prob_of_five = math.log(self.fives[1] / (self.ones[1] + self.fives[1]))

            for word in words:
                processed_word = self.process_word(word)

                if processed_word:

                    # +1 is for laplace smoothing
                    prob_of_one += math.log((self.ones[0].get(processed_word, 0) + 1) / (self.ones[1] + vocab_size))
                    prob_of_five += math.log((self.fives[0].get(processed_word, 0) + 1) / (self.fives[1] + vocab_size)) 

            predictions.append("1" if prob_of_one > prob_of_five else "5")

        return predictions
                
    # def classify(self, lines):
    #     predictions = []
    #     vocab_size = len(set(self.ones[0].keys()).union(set(self.fives[0].keys())))

    #     for line in lines:
    #         words = word_tokenize(line.split("|")[2])

    #         prob_of_one = math.log(self.ones[1] / (self.ones[1] + self.fives[1]))
    #         prob_of_five = math.log(self.fives[1] / (self.ones[1] + self.fives[1]))

    #         processed_words = []

    #         for word in words:
    #             processed_word = self.process_word(word)

    #             if processed_word:
    #                 processed_words.append(processed_word)
                    
    #         for bigram in bigrams(processed_words):

    #             # +1 is for laplace smoothing
    #             # prob_of_one += math.log((self.ones[0].get(bigram, 0) + 1) / self.ones[1])
    #             # prob_of_five += math.log((self.fives[0].get(bigram, 0) + 1) / self.fives[1]) 
                
    #             prob_of_one += math.log((self.ones[0].get(bigram, 0) + 1) / (self.ones[1] + vocab_size))
    #             prob_of_five += math.log((self.fives[0].get(bigram, 0) + 1) / (self.fives[1] + vocab_size))


    #         predictions.append("1" if prob_of_one > prob_of_five else "5")

    #     return predictions
    
