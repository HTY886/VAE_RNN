import tensorflow as tf
from ops import *
from utils import utils
import numpy as np
import os
import sys




class vrnn():
    
    def __init__(self,args,sess):
        self.sess = sess
        self.word_embedding_dim = 300
        self.num_epochs = 100
        self.num_steps = args.num_steps
        self.latent_dim = args.latent_dim
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.saving_step = args.saving_step
        self.feed_previous = args.feed_previous
        self.model_dir = args.model_dir
        self.data_dir = args.data_dir
        self.load_model = args.load
        self.lstm_length = [self.sequence_length+1]*self.batch_size
        self.utils = utils(args)
        self.vocab_size = len(self.utils.word_id_dict)
        self.KL_annealing = args.KL_annealing
        self.EOS = 0
        self.BOS = 1
        self.log_dir = os.path.join(self.model_dir,'log/')
        self.build_graph()
        
        self.saver = tf.train.Saver(max_to_keep=2)
        self.model_path = os.path.join(self.model_dir,'model_{m_type}'.format(m_type='vrnn'))
 
    def build_graph(self):
        print('starting building graph')
        
        with tf.variable_scope("input") as scope:
            self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
            self.train_decoder_sentence = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
            self.train_decoder_targets = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length)) 
            self.step = tf.placeholder(dtype=tf.float32, shape=())

            BOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.BOS
            EOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.EOS
            train_decoder_targets = tf.concat([self.train_decoder_targets,EOS_slice],axis=1)            
            train_decoder_sentence = tf.concat([BOS_slice,self.train_decoder_sentence],axis=1)
          
    
        with tf.variable_scope("embedding") as scope:
            init = tf.contrib.layers.xavier_initializer()
    
            # word embedding
            word_embedding_matrix = tf.get_variable(
                name="word_embedding_matrix",
                shape=[self.vocab_size, self.word_embedding_dim],
                initializer = init,
                trainable = True)
                 
            
            # decoder output projection    
            weight_output = tf.get_variable(
                name="weight_output",
                shape=[self.latent_dim*2, self.vocab_size],
                initializer =  init,
                trainable = True)
                
            bias_output = tf.get_variable(
                name="bias_output",
                shape=[self.vocab_size],
                initializer = tf.constant_initializer(value = 0.0),
                trainable = True)
    
            encoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix, self.encoder_inputs)
            train_decoder_sentence_embedded = tf.nn.embedding_lookup(word_embedding_matrix, train_decoder_sentence) 
        with tf.variable_scope("encoder") as scope:
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            #bi-lstm encoder
            encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=self.lstm_length,
                inputs=encoder_inputs_embedded,
                time_major=False)
    
            output_fw, output_bw = encoder_outputs
            state_fw, state_bw = state
            encoder_outputs = tf.concat([output_fw,output_bw],2)      
            self.encoder_outputs=encoder_outputs
            encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
            encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
            self.encoder_state_c=encoder_state_c
            self.encoder_state_h=encoder_state_h
        
        with tf.variable_scope("sample") as scope:
        
            w_mean = weight_variable([self.latent_dim*2,self.latent_dim*2],0.5)
            b_mean = bias_variable([self.latent_dim*2])
            scope.reuse_variables()
            b_mean_matrix = [b_mean] * self.batch_size
            
            w_logvar = weight_variable([self.latent_dim*2,self.latent_dim*2],0.5)
            b_logvar = bias_variable([self.latent_dim*2])
            scope.reuse_variables()
            b_logvar_matrix = [b_logvar] * self.batch_size
            
            mean = tf.matmul(encoder_state_h,w_mean) + b_mean
            logvar = tf.matmul(encoder_state_h,w_logvar) + b_logvar
            stddev = tf.exp(0.5 * logvar)
            noise = tf.random_normal(tf.shape(stddev))
            sampled_encoder_state_h = mean + tf.multiply(stddev,noise)
            
                
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=sampled_encoder_state_h) 
        decoder_inputs = batch_to_time_major(train_decoder_sentence_embedded, self.sequence_length+1)  
        
        with tf.variable_scope("decoder") as scope:
        
            r_num = tf.reduce_sum(tf.random_uniform([1], seed=1))
            cell = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim*2, state_is_tuple=True)
            self.cell = cell

            def train_decoder_loop(prev,i):  
                factor = tf.constant(5,shape=(),dtype=tf.float32)
                prev = tf.scalar_mul(factor,tf.add(tf.matmul(prev,weight_output),bias_output))
                prev_index = tf.nn.softmax(prev) 
                pred_prev = tf.matmul(prev_index,word_embedding_matrix)
                next_input = tf.cond(r_num > 0.6,\
                                lambda: pred_prev,\
                                lambda: decoder_inputs[i] ) #r>rate do first, else second
                return next_input

            def test_decoder_loop(prev,i):
                factor = tf.constant(5,shape=(),dtype=tf.float32)
                prev = tf.scalar_mul(factor,tf.add(tf.matmul(prev,weight_output),bias_output))
                prev_index = tf.nn.softmax(prev) 
                pred_prev = tf.matmul(prev_index,word_embedding_matrix)
                next_input = pred_prev
                return next_input

            train_decoder_output,train_decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs = decoder_inputs,
                initial_state = encoder_state,
                cell = cell,
                loop_function = train_decoder_loop,
                scope = scope
            )   
            
            #the decoder of testing
            scope.reuse_variables()
            test_decoder_output,test_decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs = decoder_inputs,
                initial_state = encoder_state,
                cell = cell,
                loop_function = test_decoder_loop,
                scope = scope
            )

            for index,time_slice in enumerate(test_decoder_output):
                test_decoder_output[index] = tf.add(tf.matmul(test_decoder_output[index],weight_output),bias_output)
                train_decoder_output[index] = tf.add(tf.matmul(train_decoder_output[index],weight_output),bias_output)
           
        
            
            test_decoder_logits = tf.stack(test_decoder_output, axis=1)
            test_pred = tf.argmax(test_decoder_logits,axis=-1)
            test_pred = tf.to_int32(test_pred,name='ToInt32')
    
            self.test_pred=test_pred
                                                           
        
    
        with tf.variable_scope("loss") as scope:
        
        
            kl_loss_batch = tf.reduce_sum( -0.5 * (logvar - tf.square(mean) - tf.exp(logvar) + 1.0) , 1)
            kl_loss = tf.reduce_mean(kl_loss_batch, 0) #mean of kl_cost over batche
            if(self.KL_annealing):
                step_scale = tf.constant(5000, dtype=tf.float32)
                kl_weight = tf.sigmoid(tf.divide(tf.subtract(self.step,step_scale),step_scale ))
                kl_loss = tf.scalar_mul(kl_loss, kl_weight)
            #kl_loss = tf.scalar_mul(tf.constant(3.0, dtype=tf.float32), kl_loss)
            self.kl_loss = kl_loss

            targets = batch_to_time_major(train_decoder_targets,self.sequence_length+1)
            loss_weights = [tf.ones([self.batch_size],dtype=tf.float32) for _ in range(self.sequence_length+1)]    #the weight at each time step
            self.loss = tf.reduce_sum(tf.contrib.legacy_seq2seq.sequence_loss(
                logits = train_decoder_output, 
                targets = targets,
                weights = loss_weights, 
                average_across_timesteps=False )) + kl_loss
            #self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
            
            #op_func = tf.train.AdamOptimizer()
            #tvars = tf.trainable_variables()
            #self.gradient = tf.gradients(self.loss, tvars) 
            #capped_grads, _ = tf.clip_by_global_norm(self.gradient, 1)
            #self.train_op = op_func.apply_gradients(zip(capped_grads, tvars))
            
            tf.summary.scalar('total_loss', self.loss)
    
    
    def train(self):
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        saving_step = self.saving_step
        summary_step = saving_step/10
        cur_loss = 0.0
        cur_kl_loss = 0.0
        
        if self.load_model:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        else:
            self.sess.run(tf.global_variables_initializer())
        step = 0
        
        for s,t in self.utils.train_data_generator(self.num_epochs):
            step += 1
            t_d = self.utils.word_drop_out(t)
            feed_dict = {
                self.encoder_inputs:s,\
                self.train_decoder_sentence:t_d,\
                self.train_decoder_targets:t, \
                self.step:step
            }
            _,loss,kl_loss = self.sess.run([self.train_op, self.loss, self.kl_loss], feed_dict)
            cur_loss += loss
            cur_kl_loss += kl_loss
            if step%(summary_step)==0:
                print('{step}: total_loss: {loss} kl: {kl_loss}'.format(step=step,loss=cur_loss/summary_step, kl_loss=cur_kl_loss/summary_step))
                cur_loss = 0.0
                cut_kl_loss = 0.0
            if step%saving_step==0:
                self.saver.save(self.sess, self.model_path, global_step=step)
            if step>=self.num_steps:
                break
                
                
    def stdin_test(self):
        sentence = 'hi'
        print(sentence)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
		
        while(sentence):
            sentence = sys.stdin.readline().lower()
            sys.stdout.flush()
            input_sent_vec = self.utils.sent2id(sentence)
            #print(input_sent_vec)
            sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32)
            sent_vec[0] = input_sent_vec
            t = np.ones((self.batch_size,self.sequence_length),dtype=np.int32)
            feed_dict = {
                    self.encoder_inputs:sent_vec,\
                    self.train_decoder_sentence:t
            }
            preds = self.sess.run([self.test_pred],feed_dict)
            pred_sent = self.utils.id2sent(preds[0][0])
            print(pred_sent)   
            
            
    def test(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))        
        step = 0
        cur_loss = 0.0
        cur_kl_loss = 0.0
        for s,t in self.utils.test_data_generator():
            step += 1
            t = np.ones((self.batch_size,self.sequence_length), dtype=np.int32)
            t_d = s
            feed_dict = {
                self.encoder_inputs:s,\
                self.train_decoder_targets:t_d,\
                self.train_decoder_sentence:t
            }
            loss,kl_loss = self.sess.run([self.loss, self.kl_loss], feed_dict)
            cur_loss += loss
            cur_kl_loss += kl_loss
            
        print('total loss: ' + str(cur_loss/step))
        print('kl divergence: ' + str(cur_kl_loss/step))
