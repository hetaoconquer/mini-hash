import numpy as np
import dataio as dio
import rnn
import  os
from datetime import datetime
import utils

TRAIN = True
train_data = dio.get_train_data()
rand_data = train_data.copy()
train_label = dio.get_train_gt()

test_data = dio.get_test_data()
test_label = dio.get_test_gt()

mean_ = np.mean(train_data, axis=0)
rand_data -= mean_

model = rnn.RNNTheano()
if TRAIN:
    EPOCH  = 1
    n = train_data.shape[0] / model.batch_size
    j = 0
    for it in xrange(EPOCH):
        np.random.shuffle(rand_data)
        for i in xrange(n):
            dt = rand_data[i*model.batch_size:(i+1)*model.batch_size]
            rt = model.sgd_step(dt)
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("%s  iterators :%6d, loss : %f  " % (time, j, rt[0]))
            j += 1
    if os.path.exists('models')==False:
        os.makedirs("models")
    utils.save_model_parameters_theano('models/models.npz',model)
utils.load_model_parameters_theano('models/models.npz',model)
d = train_data - mean_
p = model.predict(d)
B = utils.num2bit(p)
test_data -= mean_
query = model.predict(test_data)
query_b = utils.num2bit(query)
print "start calculate map ..."
print utils.cat_map(B,train_label,query_b,test_label)
