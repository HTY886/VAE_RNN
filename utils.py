import os
import random
import json
import numpy as np

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


class utils():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.sequence_length = args.sequence_length
        self.word_id_dict = read_json(os.path.join(self.data_dir,'word_id.json'))
        self.unknown_id = len(self.word_id_dict)
        self.word_id_dict['__UNK__'] = len(self.word_id_dict)
        self.droptout_id = len(self.word_id_dict)
        self.word_id_dict['__DROPOUT__'] = len(self.word_id_dict)
        self.EOS_id = 0
        self.BOS_id = 1

        self.id_word_dict = [[]]*len(self.word_id_dict)
        print(len(self.id_word_dict))
        for word in self.word_id_dict:
            self.id_word_dict[self.word_id_dict[word]] = word


    def word_drop_out(self,sents,rate=0.3):
        sents = np.array(sents)
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                if random.random()<=rate and sents[i][j]!=0:
                    sents[i][j] = self.word_id_dict['__DROPOUT__']
        return sents 


    def sent2id(self,sent,l=None):
        sent_list = sent.strip().split()
        vec = np.zeros((self.sequence_length),dtype=np.int32)
        sent_len = len(sent_list)
        unseen = 0
        for i,word in enumerate(sent_list):
            if i==self.sequence_length:
                break
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
            else:
                vec[i] = self.unknown_id
        if l:
            return vec,sent_len
        else:
            return vec  


    def id2sent(self,ids):
        word_list = []
        for i in ids:
            word_list.append(self.id_word_dict[i])
        return ' '.join(word_list)


    def train_data_generator(self,num_epos):
        for _ in range(num_epos):
            f=open('/home_local/htsungy/corpus/open_subtitles.txt','r')
            data=f.readlines()
            random.shuffle(data)
            """
            data_shuffled=[];
            for i in range(len(data)):
                if i%2 == 0 & i<500000:
                    data_shuffled.append([data[i],data[i+1]])
            random.shuffle(data_shuffled)
            """
            batch_s=[];batch_t=[];
                    
            for i in range(len(data)):
                s_sent,s_l = self.sent2id(data[i],1)
                t_sent,t_l = self.sent2id(data[i],1)
                if s_l<=30 and t_l<=30 and random.random()<0.8:
                    batch_s.append(s_sent)
                    batch_t.append(t_sent)
                    if len(batch_s)==self.batch_size:
                        yield batch_s,batch_t
                        batch_s=[];batch_t=[];


                         