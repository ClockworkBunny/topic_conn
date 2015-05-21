"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

####changing gpu device as 1
######################################
mFileName = 'C:\\Users\\rzhao001\\.theanorc.txt'
fileRead = open(mFileName, 'r')
lines = fileRead.readlines()
fileRead.close()

filewrite = open(mFileName,'w')

for line in lines:
    if 'device' in line.split():
        modif_line = line.split()
        modif_line[2] = 'gpu1\n'
        line = ' '.join(modif_line)     
    filewrite.write(line)
filewrite.close()
####changing gpu device
#######################################

import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import os
#import modelstest
#os.chdir(os.path.dirname(__file__))


warnings.filterwarnings("ignore")   



    

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,datasets_weights,
                   U, U_Topical,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   use_valid_set=True,
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    U_Topical.dtype = "float32"
    (num_topics,topic_dim) = U_Topical.shape
    word_w = img_w
    img_w = int(img_w + num_topics*topic_dim)
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:  
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))   # 100  1  3 300
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))   # size of words samples one
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    #print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    x_topic = T.tensor3('x_topic')
    Words = theano.shared(value = U, name = "Words")
    Topics = theano.shared(value=U_Topical,name="Topics")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(word_w, dtype='float32')
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    layer0_input_words = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))  
    layer0_inputs_topics = []
    for i in range(num_topics):
        sin_topic = x_topic[:,:,i]
        Topic = Topics[i].reshape((1,Topics[i].shape[0]))
        weights = sin_topic.flatten()
        weights = weights.reshape((weights.shape[0],1))
        layer0_inputs_topics.append(T.dot(weights, Topic))
    layer0_input_topics = T.concatenate(layer0_inputs_topics,1)
    layer0_input_topics = layer0_input_topics.reshape((x_topic.shape[0],1,x_topic.shape[1],num_topics*topic_dim))
    layer0_input = T.concatenate([layer0_input_words,layer0_input_topics],3)                                 
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]   #params are model parameters
        params += [Topics]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        random_index = np.random.permutation(np.arange(datasets[0].shape[0])) 
        random_index.astype('int32')
        train_set = datasets[0][random_index,:]
        train_set_weights = datasets_weights[0][random_index,:,:]
        extra_data = train_set[:extra_data_num]
        extra_data_weights = train_set_weights[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
        new_data_weights = np.append(datasets_weights[0],extra_data_weights,axis = 0)
    else:
        new_data = datasets[0]
        new_data_weights = datasets_weights[0]
    random_index = np.random.permutation(np.arange(new_data.shape[0])) 
    random_index.astype('int32')
    new_data = new_data[random_index]
    new_data_weights = new_data_weights[random_index]
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
  
    test_set_x = np.asarray(datasets[1][:,:img_h] ,"float32")
    test_set_x_topic = np.asarray(datasets_weights[1][:,:img_h,:] ,"float32")
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    if use_valid_set:
        train_set = new_data[:n_train_batches*batch_size,:]
        train_set_weights = new_data_weights[:n_train_batches*batch_size,:,:]
        val_set = new_data[n_train_batches*batch_size:,:]
        val_set_weights = new_data_weights[n_train_batches*batch_size:,:,:]     
        train_set_x, train_set_x_topic, train_set_y = shared_dataset((train_set[:,:img_h],train_set_weights,train_set[:,-1]))
        val_set_x, val_set_x_topic, val_set_y = shared_dataset((val_set[:,:img_h],val_set_weights,val_set[:,-1]))
        n_val_batches = n_batches - n_train_batches
        val_model = theano.function([index], classifier.errors(y),
                givens={
                x: val_set_x[index * batch_size: (index + 1) * batch_size],
                x_topic: val_set_x_topic[index * batch_size: (index + 1) * batch_size],
                y: val_set_y[index * batch_size: (index + 1) * batch_size]})
    else:
        train_set = new_data[:,:]    
        train_set_x, train_set_x_topic, train_set_y = shared_dataset((train_set[:,:img_h],train_set_weights,train_set[:,-1]))  
            
    #make theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                x_topic: train_set_x_topic[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            x_topic: train_set_x_topic[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
   
                

    test_layer0_input_words = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))  
    test_layer0_inputs_topics = []
    for i in range(num_topics):
        sin_topic = x_topic[:,:,i]
        Topic = Topics[i].reshape((1,Topics[i].shape[0]))
        weights = sin_topic.flatten()
        weights = weights.reshape((weights.shape[0],1))
        test_layer0_inputs_topics.append(T.dot(weights, Topic))
    test_layer0_input_topics = T.concatenate(test_layer0_inputs_topics,1)
    test_layer0_input_topics = test_layer0_input_topics.reshape((test_size,1,img_h,num_topics*topic_dim))
    test_layer0_input = T.concatenate([test_layer0_input_words,test_layer0_input_topics],3) 



    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)

    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,x_topic,y], test_error)   
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0 
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        
        print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))
        
        if val_perf >= best_val_perf:
            params_conv = [] 
            params_output = {}
            test_loss = test_model_all(test_set_x,test_set_x_topic, test_set_y) 
            test_perf = 1- test_loss 
            best_val_perf = val_perf 
            for conv_layer in conv_layers:
                params_conv.append(conv_layer.get_params())
                params_output = classifier.get_params()
                word_vec = Words.get_value()
                Topic_vec = Topics.get_value()
            print "testing" 
    test_loss = test_model_all(test_set_x,test_set_x_topic, test_set_y) 
    test_final = 1- test_loss
    return test_perf, test_final, [params_conv, params_output, word_vec,Topic_vec]


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_x_topic, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_x_topic = theano.shared(np.asarray(data_x_topic,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_x_topic, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words') and (param.name != 'Topics'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, topic_weights, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    t = []
    num_topics = len(topic_weights[0])
    pad = filter_h - 1
    zero_weights = [0.0]*num_topics
    for i in xrange(pad):
        x.append(0)
        t.append(zero_weights)
    words = sent.split()
    for index, word in enumerate(words):
        if word in word_idx_map:
            x.append(word_idx_map[word])
            t.append(topic_weights[index])
    while len(x) < max_l+2*pad:
        x.append(0)
        t.append(zero_weights)
    return x,t

def make_idx_data_cv(revs, lda_weights, word_idx_map, cv, max_l=56, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    train_weight, test_weight = [], []
    for index, rev in enumerate(revs): 
        sent,sent_weight = get_idx_from_sent(rev["text"], lda_weights[index], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"] == cv:            
            test.append(sent)
            test_weight.append(sent_weight)       
        else:  
            train.append(sent)
            train_weight.append(sent_weight)   
    train = np.array(train, dtype="int")
    train_weight = np.array(train_weight, dtype="float32")
    test_weight = np.array(test_weight, dtype="float32")
    test = np.array(test, dtype="int")
    return [train, test], [train_weight,test_weight]


if __name__=="__main__":
    
    print "loading data..."
    x = cPickle.load(open("mr_10fold.p","rb"))
    revs, W, W_Topic, LDAFilter, word_idx_map, dictionary, max_l = x[0], x[1], x[2], x[3], x[4],x[5], x[6]
    print "data loaded!"
    #mode= sys.argv[1]
    #word_vectors = sys.argv[2] 
    #num_epoch = int(sys.argv[3])
    word_vectors = "-word2vec"
    num_epoch = 30  
    non_static=True
    
    execfile("conv_net_classes.py") 
    U = np.array(W, dtype=theano.config.floatX)
    U_Topic = np.array(W_Topic, dtype=theano.config.floatX)
    print "Epoching Num: %d"%num_epoch
    results = []
    results_our = []
    r = range(0,10)
    for i in r:
        datasets, datasets_weights = make_idx_data_cv(revs, LDAFilter, word_idx_map, i, max_l=max_l,k=300, filter_h=5)
                
        perf,perf1, models = train_conv_net(datasets,datasets_weights,
                                  U, U_Topic,
                                  lr_decay=0.95,
                                  filter_hs=[3,4,5],
                                  conv_non_linear="relu",
                                  hidden_units=[100,2], 
                                  use_valid_set=True, 
                                  shuffle_batch=True, 
                                  n_epochs=num_epoch, 
                                  sqr_norm_lim=9,
                                  non_static=non_static,
                                  batch_size=100, 
                                  dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf1)
           
     
  