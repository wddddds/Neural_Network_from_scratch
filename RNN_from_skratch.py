import csv
import itertools
import nltk
import numpy as np

from RNN_utils import RNN

vocabulary_size = 10000
unknown_token = 'unknown_token'
start_token = 'start_token'
end_token = 'end_token'

# with open('/Users/kaka/Desktop/AI/NN_1/reddit-comments-2015-08.csv') as f:
#     reader = csv.reader(f, skipinitialspace=True)
#     reader.next()
#     i = 0
#     for c in reader:
#         if i <= 100:
#             i += 1
#             print c


with open('/Users/kaka/Desktop/AI/NN_1/reddit-comments-2015-08.csv') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    sentence = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentence = ['%s %s %s' % (start_token, s, end_token) for s in sentence]

tokenize_sentence = [nltk.word_tokenize(s) for s in sentence]
# print tokenize_sentence[0:100]

word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentence))


vocabulary = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocabulary]
print index_to_word
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
# print word_to_index

for i, sent in enumerate(tokenize_sentence):
    tokenize_sentence[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenize_sentence])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenize_sentence])
print X_train
print y_train


np.random.seed(42)
model = RNN(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print o.shape
print o

predictions = model.predict(X_train[10])
print predictions.shape
print predictions


# # To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
# grad_check_vocab_size = 100
# np.random.seed(42)
# model = RNN(grad_check_vocab_size, 10, bptt_truncate=1000)
# model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])


# np.random.seed(42)
# model = RNN(vocabulary_size)
# model.numpy_sdg_step(X_train[10], y_train[10], 0.005)


np.random.seed(42)
# Train on a small subset of the data to see what happens
losses = model.sgd_train(X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
