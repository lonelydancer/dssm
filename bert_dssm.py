# coding:utf-8
import os, sys, random
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
import data_helper


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#tf.compat.v1.enable_eager_execution()

class BertDssm(object):
    def __init__(self,
                 train_corpus_path=None,
                 test_corpus_path=None,
                 model_path=None,
                 query_encoder_path=None,
                 doc_encoder_path=None,
                 n_samples=10000,
                 batch_size=128,
                 buffer_size=100,
                 NEG=10,
                 epochs=5,
                 docs_max=10,
                 doc_hidden_dim=512,
                 semantic_dim=128):
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path
        self.model_path = model_path
        self.query_encoder_path = query_encoder_path
        self.doc_encoder_path = doc_encoder_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.NEG = NEG
        self.epochs = epochs
        self.docs_max = docs_max
        self.doc_hidden_dim = doc_hidden_dim
        self.semantic_dim = semantic_dim
        self.steps_per_epoch = self.n_samples // self.batch_size + 1
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

    def base_model(self, query_input, query_sem, doc_layer, doc_layer1, idx):
# print (query_input)
        doc_input = tf.keras.layers.Input(shape=(768, ), name="doc_input_%d"%(idx))
#      print (doc_input)
        # query
#doc_sem =tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="doc_sem_%s"%(idx))(doc_input)
#doc_sem = tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="doc_sem")(doc_input)
        doc_sem = doc_layer(doc_input)
        doc_sem = doc_layer1(doc_sem)
        cosine = tf.keras.layers.dot([query_sem, doc_sem], axes=1, normalize=True)
        model = tf.keras.models.Model(inputs=[query_input, doc_input], outputs=cosine)
        return model


    def create_model(self):
        '''
        query_input = tf.keras.layers.Input(shape=(768, ), name="query_input")
        pos_doc_input = tf.keras.layers.Input(shape=(768, ), name="pos_doc_input")
        neg_docs_input = [tf.keras.layers.Input(shape=(768, ), name="neg_doc_input_%s"%j) for j in range(self.NEG)]
        '''

        query_input = tf.keras.layers.Input(shape=(768, ), name="query_input")
        query_sem1 = tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="query_sem1")(query_input)
        query_sem = tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="query_sem2")(query_sem1)

        doc_layer1 =tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="doc_sem1")
        doc_layer =tf.keras.layers.Dense(self.semantic_dim, activation="tanh", name="doc_sem")

        query_pos_doc_cosine_model = self.base_model(query_input, query_sem, doc_layer, doc_layer1, 0)
        query_neg_docs_cosine_model_list = [self.base_model(query_input, query_sem, doc_layer, doc_layer1, i+1) for i in range(self.NEG)]

        query_pos_doc_cosine = query_pos_doc_cosine_model.output
        query_neg_docs_cosine = [query_neg_docs_cosine_model_list[i].output for i in range(self.NEG)]


        concat_cosine = tf.keras.layers.concatenate([query_pos_doc_cosine] + query_neg_docs_cosine)
        concat_cosine = tf.keras.layers.Reshape((self.NEG + 1, 1))(concat_cosine)
#print (concat_cosine)

        # gamma
        weight = np.array([1]).reshape(1, 1, 1)
        with_gamma = tf.keras.layers.Conv1D(1, 1, padding="same", input_shape=(self.NEG + 1, 1), activation="linear", use_bias=False, weights=[weight])(concat_cosine)
        with_gamma = tf.keras.layers.Reshape((self.NEG + 1, ))(with_gamma)

        # softmax
        prob = tf.keras.layers.Activation("softmax")(with_gamma)
#print (prob)
        _, pos_doc_input = query_pos_doc_cosine_model.input
#        print ('query_input', type(query_input))
#        print ('query_input name', query_input.name)
#        print ('pos_doc_input', type(pos_doc_input))
#        print ('doc_input name', pos_doc_input.name)
        neg_docs_input = [i.input[1] for i in query_neg_docs_cosine_model_list]
        x= [query_input, pos_doc_input] + neg_docs_input 
# print (x)
        # model
        model = tf.keras.models.Model(inputs=[query_input, pos_doc_input] + neg_docs_input, outputs=prob)
#model = tf.keras.models.Model(inputs=[query_input, pos_doc_input] + neg_docs_input, outputs=pos_doc_sem)
#model.summary()
        tf.keras.utils.plot_model(model, to_file='dssm.png', show_shapes=True)
        self.query_sem = query_sem
#        x = tf.keras.backend.print_tensor(query_sem, message="query_sem is")
#tf.keras.backend.eval(query_sem)
        return model


    def train(self):
#model, query_encoder, doc_encoder = self.create_model()
        model = self.create_model()
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy","top_k_categorical_accuracy"])
# model.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        self.steps_per_epoch2 =  957.0 // self.batch_size + 1
        model.fit_generator(data_helper.train_input_fn(fin=self.train_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=self.batch_size,
                                                       NEG=self.NEG),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            validation_data=data_helper.train_input_fn(fin=self.test_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=self.batch_size,
                                                       NEG=self.NEG),
                            validation_steps = self.steps_per_epoch2
                            )
        '''

        model.fit_generator(data_helper.test_input_fn(fin=self.train_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=self.batch_size,
                                                       NEG=self.NEG),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            validation_data=data_helper.test_input_fn(fin=self.test_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=self.batch_size,
                                                       NEG=self.NEG),
                            validation_steps = self.steps_per_epoch2
                            )
        '''
#    tf.keras.backend.eval(self.query_sem)
        model.save(self.model_path)


    def predict(self, model):
#model2 =  model.

        validation_data=data_helper.test_input_fn(fin=self.test_corpus_path,
                                                       buffer_size=self.buffer_size,
                                                       batch_size=1,
                                                       NEG=self.NEG),
        right = 0.0
        total = 0.0
        for k in validation_data:
            print ('input', k)
            result = model.predict(k, steps=957)
            for j in result:
                print ('j',j)
            res = np.argmax(result,axis=1)
            right = np.sum(res==0)
            total = result.shape[0]
        print ('right', right)
        print ('total', total)
        print ('acc', right/total)
        
if __name__ == "__main__":
    train_file = sys.argv[1]
#test_file = 'test.record'
    test_file = 'record.test'
#    test_file = 'record'
    n_sample = 50000
#n_sample = 2000000
    model_path = sys.argv[2]
    bd = BertDssm(train_corpus_path=sys.argv[1], test_corpus_path=test_file, model_path=model_path,NEG=4,n_samples=n_sample, batch_size=64, epochs=50)
# bd = BertDssm(train_corpus_path=sys.argv[1], test_corpus_path=test_file, model_path='test_model',NEG=4,n_samples=957, batch_size=64, epochs=20)
#bd.train()
#    sys.exit(0)
    model = tf.keras.models.load_model(bd.model_path)

    bd.predict(model)
