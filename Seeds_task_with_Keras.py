import utils
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.utils.np_utils import to_categorical


# np.random.seed(90)
# create model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='sigmoid'))
model.add(Dense(7, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


filename = '/Users/kaka/Desktop/AI/NN_1/seeds_data.csv'
data_set = utils.load_csv(filename)
utils.str_column_to_int(data_set, len(data_set[0])-1)

# Normalize the input variables
min_max = utils.data_set_min_max(data_set)
utils.normalize_data_set(data_set, min_max)
data_set = np.array(data_set)

print data_set
X = data_set[:, 0:7]
y = data_set[:, 7]

print X
print y
model.compile(loss='categorical_crossentropy', optimizer='Nadam',
              metrics=['mean_squared_error', 'acc', metrics.mae, metrics.categorical_accuracy])


model.fit(X, y, epochs=150, batch_size=5)

preds = model.predict(X)
print preds


# evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
