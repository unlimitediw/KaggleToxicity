'''
learn from andrew
'''
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sub = pd.read_csv('./data/sample_submission.csv')

train['comment_text'] = train['comment_text'].astype(str)
test['comment_text'] = test['comment_text'].astype(str)

# remove not general char
punct_mapping = {"_":" ","`":" "}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text

train['comment_text'] = train['comment_text'].apply(lambda x:clean_special_chars(x,punct,punct_mapping))
print('t_clean_finish')
test['comment_text'] = test['comment_text'].apply(lambda x:clean_special_chars(x,punct,punct_mapping))

print('clean_finish')
# collect all text to build the w2v
full_text = list(train['comment_text'].values) + list(test['comment_text'].values)

# tokenize
'''
    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...
'''
tk = Tokenizer(lower=True, filters = '',num_words=90000)
tk.fit_on_texts(full_text)

embedding_path1 = './embedding/crawl-300d-2M.vec'
embedding_path2 = './embedding/glove.840B.300d.txt'

embed_size = 300
max_features = 100000

# *表示接受一个元组解包，**表示接受一个字典解包
def get_coefs(word,*arr):
    return word, np.asarray(arr,dtype = 'float32')

def build_matrix(embedding_path,tokenizer):
    embeeding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
    word_index = tk.word_index
    print(len(word_index))
    # base on the total num of comment_text words
    nb_words = min(max_features,len(word_index))
    # max 100000*300
    embedding_matrix = np.zeros((nb_words+1,embed_size))
    a = 0
    for word, i in word_index.items():
        a+= 1
        if a % 1000 == 0: print(a)
        if i >= max_features:
            continue
        # it is a dict for word and it's index
        embedding_vector = embeeding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# embedding path include the prebuild vectors file
embedding_matrix = np.concatenate([build_matrix(embedding_path1,tk),build_matrix(embedding_path2,tk)],axis = -1)

print(embedding_matrix.shape)