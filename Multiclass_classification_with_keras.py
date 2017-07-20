import numpy as np
import utils
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold

np.random.seed(42)

# Load data


# load dataset
# data_frame = pd.read_csv('/Users/kaka/Desktop/AI/NN_1/iris.data', header=None)
# data_set = data_frame.values
# X = data_set[:, 0:4].astype(float)
# y = data_set[:, 4]

filename = '/Users/kaka/Desktop/AI/NN_1/seeds_data.csv'
data_set = utils.load_csv(filename)
utils.str_column_to_int(data_set, len(data_set[0])-1)

# Normalize the input variables
min_max = utils.data_set_min_max(data_set)
utils.normalize_data_set(data_set, min_max)
data_set = np.array(data_set)

X = data_set[:, 0:7]
y = data_set[:, 7]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)


# Define the baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(5, input_dim=7, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print 'Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100)

