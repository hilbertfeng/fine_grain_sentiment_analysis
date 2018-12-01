'''
@author:Hilbert
@datetime:2018-11-29 20:03
@version:1.0
@description:
使用jieba对中文信息进行分词，并用LSTM对用户评论信息
进行细粒度用户评论情感分析，分析用户是正面情绪还是负面情绪，
或者中立情绪，然后计算权值。

1 epoches 可到 0.45

'''

import jieba
import jieba.analyse
import codecs as cd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,Dense
from keras.layers import LSTM


def getText():
    #获取评论内容输入
    contents = []
    words = []
    with cd.open('../train_content.csv',encoding='utf-8',mode='rU')as f:
        for line in f:
            contents.append(line)
    print('len of contents : ',len(contents))

    with cd.open('../train_word.csv',encoding='utf-8',mode='rU')as f:
        for line in f:
            words.append(line)
    print('len of words : ',len(words))

    return contents,words


def calWordCover(contents,words,k=10):
    '''

    :param contents:
    :param words:
    :param k:
    :return:
    :param contents:
    :param words:

    '''
    keysLen = 0
    coverLen = 0
    print('关键词  /  被覆盖关键词')
    for i in range(20000):
        tags = jieba.lcut(contents[i].strip(),cut_all=False)
        keys = set([i for i in words[i].strip().split(';') if len(i) > 0])
        keysLen += len(keys)
        coverLen += len(keys & set(tags))
        if i < 5:
            print(keys,keys & set(tags))
        contents[i] = ' '.join(list(tags))
        words[i] = list(keys)
    print('情感关键词覆盖率',keysLen / coverLen)
    return contents,words


contents,words = getText()
contents,words = calWordCover(contents,words,15)


dfWords = pd.DataFrame(words)
print('平均情感关键词数量 : ',dfWords.count(axis=1).mean())
indexWord = dfWords.index[~pd.isnull(dfWords.iloc[:,0])]
word = dfWords.iloc[indexWord,0].tolist()
contents = pd.DataFrame(contents).iloc[indexWord,0].tolist()
tokenizer = Tokenizer(num_words=32000)
tokenizer.fit_on_texts(contents+word)
sequences = tokenizer.texts_to_sequences(contents)
sequences_words = tokenizer.texts_to_sequences(word)
data_x = pad_sequences(sequences,maxlen=50)


def getDataY(data_x):
    '''

    :param data_x:
    :return:
    numpy返回一个数组data_x

    '''

    data_y = []
    for i in range(data_x.shape[0]):
        try:
            data_y.append(list(data_x[i]).index(sequences[i][0]))
        except:
            data_y.append(-1) #如果情感关键词不在内容中
    return np.array(data_y)



data_y = getDataY(data_x)
onehot_y = to_categorical(data_y[data_y >= 0],num_classes=50)
train_x,test_x,train_y,test_y = train_test_split(data_x[data_y >= 0],onehot_y)



def trainModel(train_x,test_x,train_y,test_y):
    '''

    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :return:
    model

    '''
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,128))
    model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(50,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_x,train_y,
              batch_size=32,  #设置batch_size时要根据自己配置
              epochs=1,  #设置轮式越高所需的时间越长
              validation_data=(test_x,test_y))

    return model



model = trainModel(train_x,test_x,train_y,test_y)

def getPredWord(model,word_index,x,y):
    '''
    计算预测情感关键词
    :param model:
    :param word_index:
    :param x:
    :param y:
    :return:
    '''
    pos_pred = model.predict_classes(x)
    pos_y = np.argmax(y,axis=1)
    index2word = pd.DataFrame.from_dict(word_index,'index').reset_index().set_index(0)

    print('模型预测关键词')
    print(index2word.loc[x[[i for i in range(pos_pred.shape[0])],pos_pred]].head(5))
    print('实际样本关键词')
    print(index2word.loc[x[[i for i in range(pos_y.shape[0])],pos_y]].head(5))


getPredWord(model,tokenizer.word_index,test_x,test_y)
