from collections import Counter
import math



class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1e-5):
        self.alpha = alpha
        self.mockup: dict = {
            "labels": {},
            "words": {},
        }

    def fit(self, X, y):
        """ Fit Naive Bayes classifier according to X, y."""
        collection = []
        for title, tag in zip(X, y):
            for word in title.split():
                couple = (word, tag)
                collection.append(couple)

        self.uncommon_words = Counter(collection)
        print(f"uncommon_words", {self.uncommon_words})

        self.count_dictionary = dict(Counter(y))
        print(f"count_dictionary", {self.count_dictionary})

        words = [word for title in X for word in title.split()]
        self.count_words = dict(Counter(words))
        print(f"count_words", {self.count_words})

        self.mockup = {
            "labels": {},
            "words": {},
        }

        for publish in self.count_dictionary:
            counter = 0
            for word, tag_name in self.uncommon_words:
                if publish == tag_name:
                    counter += self.uncommon_words[(word, publish)]
            c_d = self.counted_dict[publish]
            params = {
                "tag_count": counter,
                "probability": c_d / len(y),
            }
            self.mockup["tags"][publish] = params

        for word in self.count_words:
            params = {}
            for publish in self.count_dict:
                vk = self.mockup["tags"][publish]["tags_count"]
                vik = self.uncommon_words.get((word, publish), 0)
                lenght = len(self.count_words)
                alpha = self.alpha
                smooth = (vik + alpha) / (vk + vik * lenght)
                params[publish] = smooth
            self.mockup["words"][word] = params

    def predict(self, X):
        """ Perform classification on an array of test vectors X. """
        words = X.split()
        fate = []
        for current_tag in self.mockup["labels"]:
            probability = self.mockup["labels"][current_tag]["probability"]
            total = math.log(probability, math.e)
            for word in words:
                word_dict = self.mockup["words"].get(word, None)
                if word_dict:
                    total += math.log(word_dict[current_tag], math.e)
            fate.append((total, current_tag))
        _, forecast = max(fate)
        return forecast

    def score(self, X_test, y_test):
        """ Returns the mean accuracy on the given test data and labels. """
        correct = []
        for element in X_test:
            correct.append(self.predict(element))
        final = sum(0 if correct[i] != y_test[i] else 1 for i in range(len(X_test))) / len(X_test)

        return final