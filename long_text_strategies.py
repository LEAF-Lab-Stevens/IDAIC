import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import numpy as np

class LongTextStrategy:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def preprocess(self, x, y):
        truncated_x, truncated_y = self.long_text_handle(x, y)
        return truncated_x, truncated_y

class CustomTruncationStrategy(LongTextStrategy):
    def __init__(self, max_length: int = 512):
        super().__init__(max_length)

    def long_text_handle(self, x, y):
        truncated_x = [self._custom_truncate_sequence(sequence) for sequence in x]
        return truncated_x, y

    def _custom_truncate_sequence(self, sequence):
        word_list = sequence.split(' ')
        if len(word_list) <= self.max_length:
            return " ".join(word_list)
        else:
            start = int((len(word_list) - self.max_length) / 2)
            end = int((len(word_list) + self.max_length) / 2)
            return " ".join(word_list[start:end])


class TextSummarizationStrategy(LongTextStrategy):
    def __init__(self) -> None:
        super().__init__()

    def long_text_handle(self, x, y):
        truncated_x = self.text_summarization(x)
        return truncated_x, y

    def text_summarization(self, x):
        x_summary = []
        for text in x:
            if len(text.split(" ")) < 512:
                x_summary.append(text)
            else:
                summary = self._generate_summary(text)
                x_summary.append(summary)
        return x_summary

    def _generate_summary(self, text: str) -> str:
        sentences = nltk.sent_tokenize(text)
        sentence_scores = {}

        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word.isalnum():
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = 1
                    else:
                        sentence_scores[sentence] += 1

        if len(sentence_scores) == 0:
            avg_score = 0
        else:
            avg_score = sum(sentence_scores.values()) / len(sentence_scores)

        summary = ''
        for sentence in sentences:
            if sentence_scores.get(sentence, 0) > avg_score:
                summary += ' ' + sentence

        summary = ' '.join(summary.split())
        return summary

class SpaCyTextSummarizationStrategy(LongTextStrategy):
    def __init__(self):
        super().__init__()

    def long_text_handle(self, x, y):
        truncated_x = self.text_summarization(x)
        return truncated_x, y

    def text_summarization(self, x):
        x_summary = []
        for text in x:
            summary = self._generate_summary(text)
            x_summary.append(summary)
        return x_summary

    def _generate_summary(self, text: str):
        if len(text.split(" "))<=512:
            return text
        else:
            per = 0
            if len(text.split(" "))>512 and len(text.split(" "))<=1000:
                per = 0.25
            elif len(text.split(" "))>1000 and len(text.split(" "))<=2000:
                per = 0.12
            elif len(text.split(" "))>2000 and len(text.split(" "))<=3000:
                per = 0.06
            else:
                per = 0.03
            new_text = self.summarize(text, per)
            return new_text
          

    def summarize(text, per):
        nlp = spacy.load('en_core_web_sm')
        doc= nlp(text)
        tokens=[token.text for token in doc]
        word_frequencies={}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency=max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word]=word_frequencies[word]/max_frequency
        sentence_tokens= [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():                            
                        sentence_scores[sent]=word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent]+=word_frequencies[word.text.lower()]
        select_length=int(len(sentence_tokens)*per)
        summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
        final_summary=[word.text for word in summary]
        summary=''.join(final_summary)
        return summary


class WindowShiftStrategy(LongTextStrategy):
    def __init__(self, window_size, step_size):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size

    def long_text_handle(self, x, y):
        truncated_x, truncated_y, no_of_windows = self.window_shift(x, y)
        return truncated_x, truncated_y

    def window_shift(self, x, y):
        truncated_x = []
        truncated_y = []
        no_of_windows = []

        for j in range(len(x)):
            windows, labels, count = self._generate_windows(x[j], y[j])
            truncated_x.extend(windows)
            truncated_y.extend(labels)
            no_of_windows.extend(count)

        return truncated_x, truncated_y, no_of_windows

    def _generate_windows(self, sequence, label):
        windows = []
        y = []
        count = 0
        seq = sequence.split(" ")

        for i in range(0, len(seq), self.step_size):
            count += 1
            if i + self.window_size <= len(seq):
                windows.append(" ".join(seq[i:i + self.window_size]))
                y.append(label)
            else:
                windows.append(" ".join(seq[i:]))
                y.append(label)

        return windows, y, [count]

    def window_shift_reverse(self, sequence, no_of_windows):
        start = 0
        y_predicted = []

        for i in no_of_windows:
            y = int(np.average(sequence[start:start+i]))
            start = start + i
            y_predicted.append(y)

        return y_predicted

    def window_shift_reverse_prob(self, sequence, no_of_windows):
        y_predicted = []

        for i in no_of_windows:
            rows, cols = sequence.shape
            y = np.average(sequence[:i, :cols], axis=0)
            sequence = sequence[i:, :]
            y_predicted.append(y)

        return np.array(y_predicted)
