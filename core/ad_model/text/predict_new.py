import re
import jieba
import pickle
import numpy as np
from keras.preprocessing import sequence as keras_seq
from keras.models import load_model
import os
from keras import backend as K

fileroad = os.path.dirname(os.path.realpath(__file__))


def remove_num(content):
    if content == np.nan:
        return ''
    content = str(content)
    # return content
    new_content = re.sub('[0-9]*\.*[0-9]+', '', content)
    new_content = re.sub('（.*）', '', new_content)
    new_content = re.sub('\(.*\)', '', new_content)
    new_content = new_content.replace('×', '')
    new_content = new_content.replace('*', '')
    new_content = new_content.replace('mm', '')
    new_content = new_content.replace('cm', '')
    new_content = new_content.replace('%', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('\r', '')
    new_content = new_content.replace('\n', '')

    #    if new_content.find('穿刺记录') != -1:
    #        new_content = new_content[:new_content.find('穿刺记录')]
    return new_content


def load_data(tokenizer, sen):
    sen = remove_num(sen)
    sen = ' '.join(list(jieba.cut(sen)))
    text_seq = tokenizer.texts_to_sequences([sen])
    text_array = keras_seq.pad_sequences(text_seq, maxlen=300, dtype='int32', padding='pre')
    return text_array


def process_document(doc, sentence_size, tokenizer):
    sentences = re.split('[。]', doc)
    while (len(sentences) < sentence_size):
        sentences.append('')
    sentences = list(map(lambda x: ' '.join(list(jieba.cut(x))), sentences))
    sentences = tokenizer.texts_to_sequences(sentences)
    return sentences


def pad_sentence(sentences, max_len):
    sentences = keras_seq.pad_sequences(sentences, maxlen=max_len, dtype='int32', padding='post')
    return sentences


def load_data_1(tokenizer, sen):
    sen = remove_num(sen)
    sentence_size = 9
    maxlen = 30
    res = process_document(sen, sentence_size, tokenizer)
    res = pad_sentence(res, maxlen)
    return res.reshape([-1, sentence_size, maxlen])


def predict(sen, model_type):
    [tokenizer] = pickle.load(open("core/ad_model/text/token.p", "rb"))
    text_array = load_data_1(tokenizer, sen)
    model = load_model('core/ad_model/text/adtextmodel.m')
    res = model.predict(text_array)
    return res


# if __name__ == '__main__':
#     #malignant
#     print(fileroad)
#     sen = '脑中线结构未见移位。左侧颞叶、双侧额顶叶多发斑片状异常信号，T2WI FLAIR为高信号。弥散成像未见异常信号。脑室系统、脑池及脑沟形态、大小、信号未见异常。鞍区结构未见特殊。颅底结构、信号未见病理性变化。双侧颈内动脉虹吸段，床突上段，双侧大脑前、中、后动脉形态、走行、信号未见异常。未见异常供血动脉，引流静脉以及病理性血管团。'
#     # data = pd.read_csv('data.csv',index_col=0,header=None)
#     result = predict(sen, '')
#     for i in result:
#         i = i.tolist()
#         maxi = i.index(max(i))
#         print(maxi)
#         if maxi == 2:
#             print('患者可能患有老年痴呆，建议进一步诊疗')
#         elif maxi == 1:
#             print('患者患有轻度认知障碍，建议进一步诊疗')
#         elif maxi == 0:
#             print('目前身体没有痴呆症状')
# print(result)
# relist = result.tolist()
# print(relist)
# print(np.argmax(relist))
# for i in result:
#     print(i)

# print(predict(sen, ''))

def diagnosis_AD_text(sen):
    result = predict(sen, '')
    K.clear_session()
    for i in result:
        i = i.tolist()
        maxi = i.index(max(i))
        # print(maxi)
        if maxi == 2:
            return '患者可能患有老年痴呆，建议进一步诊疗'
        elif maxi == 1:
            return '患者患有轻度认知障碍，建议进一步诊疗'
        elif maxi == 0:
            return '目前身体没有痴呆症状'
