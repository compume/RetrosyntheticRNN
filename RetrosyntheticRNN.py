!ls /home/kesci/input/

import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
%matplotlib inline
import re
from sklearn.model_selection import train_test_split


train=pd.read_csv('/home/kesci/input/competition/train.txt',sep='\n')
train['rea']=train['id,reactants>reagents>production'].apply(lambda x:x.split(',')[1])
train.drop('id,reactants>reagents>production',axis=1,inplace=True)
train['rea']=train['rea'].apply(lambda x:x.split('|')[0])
train['reactants']=train['rea'].apply(lambda x:x.split('>')[0])
train['reagents']=train['rea'].apply(lambda x:x.split('>')[1])
train['production']=train['rea'].apply(lambda x:x.split('>')[2])
train.drop('rea',axis=1,inplace=True)
token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
train['rea']=train['reagents']+'>'+train['production']
import re
def tool(x):
    x=re.split(token_regex,x)
    l=[]
    for i in x:
        if i!='':
            i=re.sub(":.*?(?=])","",i)
            l.append(i)
    return ' '.join(l)
train['rea'] =train['rea'].apply(tool)
train['reactants'] =train['reactants'].apply(tool)
df=train[['rea','reactants']]
df.columns=['inputs','targets']
del train

df.targets = df.targets.apply(lambda x : 'START_ '+ x + ' _END')


all_inp_words=set()
for inp in df.inputs:
    for word in inp.split():
        if word not in all_inp_words:
            all_inp_words.add(word)
    
all_tar_words=set()
for tar in df.targets:
    for word in tar.split():
        if word not in all_tar_words:
            all_tar_words.add(word)
            
            
input_words = sorted(list(all_inp_words))
target_words = sorted(list(all_tar_words))
num_encoder_tokens = len(all_inp_words)
num_decoder_tokens = len(all_tar_words)

input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])
    

def atf_learn(batch_size,i,text):
      #  print text
        zgent=text.split('>')[0]
      #  print zgent
        zinput=text.split('>')[1]
        output=np.zeros((batch_size, 8),dtype='float32')
        if "+" in zgent:              ### 0 solvent-ions
         #  print zinput
           if "-" in zgent:
              output[i%32,0]=3
           else:
              output[i%32,0]=1
        elif "-" in zgent:
           output[i%32,0]=2
        if "=" in zgent:              ### 1 solvent-unsaturation
           if "#" in zgent:
              output[i%32,1]=3
           else:
              output[i%32,1]=1
        elif "#" in zgent:
           output[i%32,1]=2
        if ". O ." in zgent:          ### 2 solvent-protic
           output[i%32,2]=1
        elif re.match('^O \.',zgent) is not None:
           output[i%32,2]=1
        elif re.match('(.*)\. O$',zgent) is not None:
           output[i%32,2]=1
        if "( O )" in zgent or "C O ." in zgent:
           output[i%32,2]=2
        if ". N ." in zgent:
           output[i%32,2]=3
        elif re.match('^N \.',zgent) is not None:
           output[i%32,2]=3
        elif re.match('(.*)\. N$',zgent) is not None:
           output[i%32,2]=3
        if " 1 " in zgent:            ### 3 solvent-ring
           if re.match('(.*)= . . = . . =(.*)',zgent) is not None:  ### aromatic
              output[i%32,3]=1
           if " 2 " in zgent:
              if re.match('(.*)= . . = . . =(.*)',zgent) is not None: 
                 output[i%32,3]=2
              else:
                 output[i%32,3]=3
           else:
              output[i%32,3]=4

        if "+" in zinput:              ### 4 prod-ions
         #  print zinput
           if "-" in zinput:
              output[i%32,4]=3
           else:
              output[i%32,4]=1
        elif "-" in zinput:
           output[i%32,4]=2
        if "=" in zinput:              ### 5 prod-unsaturation
           if "#" in zinput:
              output[i%32,5]=3
           else:
              output[i%32,5]=1
        elif "#" in zinput:
           output[i%32,5]=2
        if ". O ." in zinput:          ### 6 prod-protic
           output[i%32,6]=1
        elif re.match('^O \.',zinput) is not None:
           output[i%32,6]=1
        elif re.match('(.*)\. O$',zinput) is not None:
           output[i%32,6]=1
        if "( O )" in zinput or "C O ." in zinput:
           output[i%32,6]=2
        if ". N ." in zinput:
           output[i%32,6]=3
        elif re.match('^N \.',zinput) is not None:
           output[i%32,6]=3
        elif re.match('(.*)\. N$',zinput) is not None:
           output[i%32,6]=3
        if " 1 " in zinput:            ### 7 prod-ring
           if re.match('(.*)= . . = . . =(.*)',zinput) is not None:  ### aromatic
              output[i%32,7]=1
           if " 2 " in zinput:
              if re.match('(.*)= . . = . . =(.*)',zinput) is not None:
                 output[i%32,7]=2
              else:
                 output[i%32,7]=3
           else:
              output[i%32,7]=4
        return output



def data_generator(batch_size): 
    while True:
        indexs = list(range(df.shape[0]))
        np.random.shuffle(indexs)
        for idx in range(0,int(len(indexs)/batch_size)):
            encoder_input_data = np.zeros((batch_size, 700),dtype='float32')
            knowledge_input_data = np.zeros((batch_size, 700),dtype='float32')    
            decoder_input_data = np.zeros((batch_size, 700),dtype='float32')
            decoder_target_data = np.zeros((batch_size, 700, num_decoder_tokens),dtype='float32')
            for i in range(idx*batch_size,batch_size*(idx+1)):
                input_text=df.inputs[i]
                target_text=df.targets[i]
                knowledge_input_data=atf_learn(batch_size,i,input_text)
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i%32, t] = input_token_index[word]
                #    knowledge_input_data[i%32, t] = input_token_index[word]
                for t, word in enumerate(target_text.split()):
                    decoder_input_data[i%32, t] = target_token_index[word]
                    if t > 0:
                        decoder_target_data[i%32, t - 1, target_token_index[word]] = 1.
            yield ([encoder_input_data,knowledge_input_data,decoder_input_data], decoder_target_data)
        #    yield ([encoder_input_data,decoder_input_data], decoder_target_data)
        

import keras
from keras.layers import Input, LSTM, Embedding, Dense,Layer, Lambda,Bidirectional,RNN,concatenate,Reshape
from keras.models import Model
from keras.utils import plot_model
#from keras.engine.topology import Merge
from keras import backend as K
import random
embedding_size = 40      # 120
knowledge_size = 8
total_size= 48


def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)


encoder_inputs = Input(shape=(None,))
encoder_inputs2 = Input(shape=(None,))   #works
en_x=Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)

en_x = Reshape((num_encoder_tokens, embedding_size))(en_x)

#model_right=Embedding(num_encoder_tokens, knowledge_size)(encoder_inputs2) 
model_right=Reshape((num_encoder_tokens, knowledge_size))(encoder_inputs2)   #works

#model_right=Lambda(antirectifier,
 #                output_shape=antirectifier_output_shape)

#added=en_x           # mod
added = concatenate([en_x,model_right],axis=2)     #works
### added = keras.layers.Add()([en_x, model_right])
### Merge([en_x,model_right], mode='concat')

encoder = LSTM(48, return_state=True)
encoder_outputs, state_h, state_c = encoder(added)

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None,))

dex=  Embedding(num_decoder_tokens, total_size)(decoder_inputs)

dex = Reshape((num_encoder_tokens, total_size))(dex)



decoder_lstm = LSTM(48, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='tanh')
#decoder_dense = Dense(num_decoder_tokens, activation='relu')

decoder_outputs = decoder_dense(decoder_outputs)

#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model = Model([encoder_inputs,encoder_inputs2, decoder_inputs], decoder_outputs) #works

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(48, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)

encoder_states = [state_h, state_c]


# Initial state.
decoder_inputs = Input(shape=(None,))

dex=Embedding(num_decoder_tokens, embedding_size)

final_dex= dex(decoder_inputs)


decoder_lstm = LSTM(48, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, encoder_inputs2, decoder_inputs], decoder_outputs)  #works
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

from keras.callbacks import TensorBoard, ModelCheckpoint
checkpoint = ModelCheckpoint('model_{epoch:02d}.h5',monitor='loss', period=5)
model.fit_generator(data_generator(50),epochs=50, samples_per_epoch=int(df.inputs.shape[0]/50),callbacks=[checkpoint])


