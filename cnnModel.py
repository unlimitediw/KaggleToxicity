'''
learn from andrew
'''
import time

import pandas as pd
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
# _BACKEND = 'tensorflow'
# from keras import backend as K # tensorflow default
# from keras import initializers, regularizers, constraints
from keras.layers import *  # Input,Embedding, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import matplotlib.pyplot as plt
# import data

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sub = pd.read_csv('./data/sample_submission.csv')

train['comment_text'] = train['comment_text'].astype(str)
test['comment_text'] = test['comment_text'].astype(str)

# remove not general char
punct_mapping = {"_": " ", "`": " "}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text


train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
print('t_clean_finish')
test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

print('clean_finish')


# collect all text to build the w2v
full_text = list(train['comment_text'].values) + list(test['comment_text'].values)

tk = Tokenizer(lower=True, filters='', num_words=90000)
tk.fit_on_texts(full_text)

# tokenize
'''
    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...
'''

'''



embedding_path1 = './embedding/crawl-300d-2M.vec'
embedding_path2 = './embedding/glove.840B.300d.txt'

# *表示接受一个元组解包，**表示接受一个字典解包
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def build_matrix(embedding_path, tokenizer):
    embeeding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
    word_index = tk.word_index
    print(len(word_index))
    # base on the total num of comment_text words
    nb_words = min(max_features, len(word_index))
    # max 100000*300
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    a = 0
    for word, i in word_index.items():
        a += 1
        if a % 1000 == 0: print(a)
        if i >= max_features:
            continue
        # it is a dict for word and it's index
        embedding_vector = embeeding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# embedding path include the prebuild vectors file
embedding_matrix = np.concatenate([build_matrix(embedding_path1, tk), build_matrix(embedding_path2, tk)], axis=-1)

df = pd.DataFrame(embedding_matrix)
df.to_csv('./embedding/embeddingMatrix')
'''
embedding_matrix = pd.read_csv('./matrix/embeddingMatrix.csv').values()

y = np.where(train['target'] >= 0.5, True, False) * 1
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white',
                    'psychiatric_or_mental_illness']
for col in identity_columns + ['target']:
    train[col] = np.where(train[col] >= 0.5, True, False)

# collect all text to build the w2v
full_text = list(train['comment_text'].values) + list(test['comment_text'].values)

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)



embed_size = 300
max_features = 100000


# baseline lstm + attention 5-fold
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizier=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        # glorot: draws samples from a uniform distribution within [-limit, limit]
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizier)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],), initializer=self.init,
                                 name='{}_b'.format(self.name), regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],), initializer='zero',
                                     name='{}_b'.format(self.name), regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        # eij cal the each layer
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            # tf.cast: cast a tensor to a new type
            a *= K.cast(mask, K.floatx())

        a = K.expand_dims(a)
        # ??
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


# Attention end here

# CNN
def build_model(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size,
                embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128,
                dr=0.1, patience=3):
    file_path = "best_model.hdf5"

    # check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    inp = Input(shape=(max_len,))
    # data reini
    x = Embedding(max_features + 1, embed_size * 2, weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)  # 0 for default
    # provide data with Attention methods
    att = Attention(max_len)(x1)
    # from benchmark kernel
    x = Conv1D(conv_size, 2, activation='relu', padding='same')(x1)
    x = MaxPool1D(5, padding='same')(x)
    x = Conv1D(conv_size, 3, activation='relu', padding='same')(x)
    x = MaxPool1D(5, padding='same')(x)
    x = Flatten()(x)
    # ?? att x not same thing, similar to res, u-net
    x = concatenate([x, att])

    x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d, metrics=["accuracy"]))
    model.fit(X_train, y_train, batch_size=128, epochs=3, validation_data=(X_valid, y_valid), verbose=2,
              callbacks=[early_stop])

    return model




# validation
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # background positive subgroup negative auc
BNSP_AUC = 'bnsp_auc'
oof_name = 'predicted_target'


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, oof_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup[oof_name])


def compute_bpsn_auc(df, subgroup, label, oof_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[oof_name])


def compute_bnsp_auc(df, subgroup, label, oof_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[oof_name])


def compute_bias_metrics_for_model(dataset,subgroups,model,label_col,include_asegs= False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size':len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset,subgroup,label_col,model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc',ascending=True)

def calculate_overall_auc(df, oof_name):
    true_labels = df['target']
    predicted_labels = df[oof_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

# training part
def train_model(X, X_test, y, tokenizer, max_len):
    oof = np.zeros((len(X), 1))
    prediction = np.zeros((len(X_test), 1))
    scores = []
    test_tokenized = tokenizer.texts_to_sequences(test['comment_text'])
    X_test = pad_sequences(test_tokenized, maxlen=max_len)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        valid_df = X_valid.copy()

        train_tokenized = tokenizer.texts_to_sequences(X_train['comment_text'])
        valid_tokenized = tokenizer.texts_to_sequences(X_valid['comment_text'])

        X_train = pad_sequences(train_tokenized, maxlen=max_len)
        X_valid = pad_sequences(valid_tokenized, maxlen=max_len)

        model = build_model(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix,
                            lr=1e-3, lr_d=0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.05,
                            patience=3)

        pred_valid = model.predict(X_valid)
        oof[valid_index] = pred_valid
        # validation part

        valid_df[oof_name] = pred_valid
        valid_df[oof_name] = pred_valid

        bias_metrics_df = compute_bias_metrics_for_model(valid_df,identity_columns,oof_name,'target')
        scores.append(get_final_metric(bias_metrics_df,calculate_overall_auc(valid_df,oof_name)))

        prediction += model.predict(X_test, batch_size=1024,verbose = 1)

        prediction /= n_fold

        return oof,prediction,scores

oof_name = 'predicted_target'
max_len = 250
oof, prediction, scores = train_model(X=train, X_test=test, y=train['target'], tokenizer=tk, max_len=max_len)
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

plt.hist(prediction);
plt.hist(oof);
plt.title('Distribution of predictions vs oof predictions')