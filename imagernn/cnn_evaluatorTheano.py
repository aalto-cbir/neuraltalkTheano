import numpy as np
import code
import theano
from theano import config
import theano.tensor as tensor
from theano.ifelse import ifelse
from collections import OrderedDict
import time
from imagernn.utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv

class CnnEvaluator:
  """ 
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params,Wemb = None):

    self.word_encoding_size = params.get('word_encoding_size', 512)
    image_feat_size = params.get('image_feat_size', 512)
    aux_inp_size = params.get('aux_inp_size', -1)

    self.n_fmaps_psz = params.get('n_fmaps_psz', 100)
    self.filter_hs = params.get('filter_hs', [])
    
    vocabulary_size = params.get('vocabulary_size',-1)
    self.sent_enc_size = params.get('sent_encoding_size',-1)# size of CNN vectors hardcoded here

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    if Wemb == None:
        model['Wemb'] = initwTh(vocabulary_size, self.word_encoding_size) # word encoder
    if params.get('swap_aux',0) == 0:
        model['WIemb'] = initwTh(image_feat_size, self.sent_enc_size) # image encoder
    else:
        model['WIemb'] = initwTh(aux_inp_size, self.sent_enc_size) # image encoder

    model['b_Img'] = np.zeros((self.sent_enc_size)).astype(config.floatX)

    model['Wfc_sent'] = initwTh(self.n_fmaps_psz * len(self.filter_hs), self.sent_enc_size) # word encoder
    model['bfc_sent'] = np.zeros((self.sent_enc_size)).astype(config.floatX)

    # Decoder weights (e.g. mapping to vocabulary)

    update_list = ['Wemb','Wfc_sent', 'bfc_sent','WIemb','b_Img']
    self.regularize = ['Wemb','Wfc_sent','WIemb']

    self.model_th = self.init_tparams(model)
    # Share the Word embeddings with the generator model
    if Wemb != None:
        self.model_th['Wemb'] = Wemb 
    self.updateP = OrderedDict()
    for vname in update_list:
        self.updateP[vname] = self.model_th[vname]

# ========================================================================================
  def init_tparams(self,params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# ========================================================================================
 # BUILD CNN evaluator forward propogation model
  def build_model(self, tparams, options):
    trng = RandomStreams()
    rng = np.random.RandomState()

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    xWi = tensor.matrix('xW', dtype='int64')
    # Now input is transposed compared to the generator!!
    xW = xWi.T
    n_samples = xW.shape[0]
    n_words= xW.shape[1]
    
    Words = tensor.concatenate([tparams['Wemb'],theano.tensor.alloc(numpy_floatX(0.),1,self.word_encoding_size)],axis=0)
    embW = Words[xW.flatten()].reshape([options['batch_size'], 1, n_words, self.word_encoding_size])
    sent_emb, cnn_out , tparams, use_noise = self.sent_conv_layer(tparams, options, embW, options['batch_size'])
    
    xI = tensor.matrix('xI', dtype=config.floatX)
    # Now to embed the image feature vector and calculate a similarity score 
    if options.get('mode','batchtrain') == 'batchtrain':
        # In train mode we compare a batch of images against each others captions.
        batch_size = options['batch_size']
    else:
        # In predict mode we compare multiple captions against a single image 
        posSamp = tensor.ivector('posSamp')
        batch_size = posSamp.shape[0] 
        
    probMatch, sim_score = multimodal_cosine_sim_softmax(xI, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    
    inp_list = [xWi, xI]
    
    if options.get('mode','batchtrain') == 'batchtrain':
        cost = -(tensor.log(probMatch.diagonal()).sum())/batch_size
    else:
        cost = -(tensor.log(probMatch[0,posSamp]).sum())/batch_size
        inp_list.append(posSamp)
    
    f_pred_sim_prob = theano.function(inp_list[:2], probMatch, name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(inp_list[:2], sim_score, name='f_pred_sim_scr')
    f_sent_emb = theano.function(inp_list[:1], cnn_out, name='f_sent_emb')

    return use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb], cost, sim_score, tparams
    
# ========================================================================================
 # BUILD CNN evaluator forward propogation model with taking direct inputs from lstm gen
  def build_advers_eval(self, tparams, options, gen_inp_list, gen_out):

    xWRefi = tensor.matrix('xWR', dtype='int64')
    xWRef = xWRefi.T 
    n_words= xWRef.shape[1]

    zero_guy = theano.tensor.alloc(numpy_floatX(0.),1,self.word_encoding_size)
    
    Words = tensor.concatenate([tparams['Wemb'],zero_guy],axis=0)
    #Words = tparams['Wemb']

    # Need to make sure that n_words >= n_gen_words, using the limit on generator
    n_gen_words = gen_out.shape[0]
    embGen = ifelse(tensor.gt(n_words, n_gen_words),tensor.concatenate([gen_out,theano.tensor.alloc(numpy_floatX(0.),n_words-n_gen_words,self.word_encoding_size)], axis=0),gen_out)
    embGen = tensor.shape_padleft(embGen, n_ones=2)
    embWRef = Words[xWRef.flatten()].reshape([options['cnn_batch_size'], 1, n_words, self.word_encoding_size])
    embW = tensor.concatenate([embWRef, embGen], axis=0) 
    
    sent_emb, cnn_out , tparams, use_noise = self.sent_conv_layer(tparams, options, embW, options['cnn_batch_size'] + 1) # +1 to include generated sample
    
    # Now to embed the image feature vector and calculate a similarity score 
    xImg = gen_inp_list[options['swap_aux']]
        
    probMatch, sim_score = multimodal_cosine_sim_softmax(xImg, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    
    inp_list = [xWRefi]
    for inp in gen_inp_list:
      if inp not in inp_list:
          inp_list.append(inp)
    #import pdb;pdb.set_trace()
    f_pred_sim_prob = theano.function(inp_list, probMatch, name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(inp_list, sim_score, name='f_pred_sim_scr')
    f_sent_emb = theano.function(inp_list, cnn_out, name='f_sent_emb')
    
    smooth_eps = tensor.constant(options['smooth_eps'],dtype=config.floatX)
    costEval = -tensor.log(probMatch[0,0]+smooth_eps)
    costGen = -tensor.log(probMatch[0,-1]+smooth_eps)

    return use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb], [costEval, costGen], sim_score, tparams
    
# ========================================================================================
  ####################################################################################
  # Defines the convolution layer on sentences.
  # -- Input is word embeddings stacked as a n_word * enc_size "image"
  # -- Filters are all of width equal to enc_size, height varies (3,4,5 grams etc.) 
  # -- Also pooling is taking max over entire filter output, i.e each filter output 
  #    is converted to a single number!
  # -- Output is stacking all the filter outputs to a single vector, 
  #    sz = (batch-size,  n_filters)
  ####################################################################################
  def sent_conv_layer(self, tparams, options, embW, batch_size):
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    trng = RandomStreams()
    rng = np.random.RandomState()
    max_sent_len = options.get('maxlen',0)
    filter_shapes = []
    self.conv_layers = []
    pool_sizes = []
    filter_w = self.word_encoding_size
    layer1_inputs = []
    for filter_h in self.filter_hs:
        filter_shapes.append((self.n_fmaps_psz, 1, filter_h, filter_w))
        if max_sent_len > 0:
            image_shape = [batch_size, 1, max_sent_len, self.word_encoding_size]
        else:
            image_shape = None
        pool_sizes.append((max_sent_len-filter_h+1, self.word_encoding_size-filter_w+1))
        conv_layer = LeNetConvPoolLayer(rng, input= embW, image_shape= image_shape, filter_shape=filter_shapes[-1], 
                                poolsize=pool_sizes[-1], non_linear=options['conv_non_linear'])
        # flatten all the filter outputs to a single vector
        cout = conv_layer.output.flatten(2)
        self.conv_layers.append(conv_layer)
        layer1_inputs.append(cout)
        self.updateP.update(conv_layer.params)
        tparams.update(conv_layer.params)
    
    layer1_input = tensor.concatenate(layer1_inputs,axis=1)
    
    # Now apply dropout on the cnn ouptut
    if options['use_dropout']:
        cnn_out = dropout_layer(layer1_input, use_noise, trng, options['drop_prob_cnn'],layer1_input.shape)
    else:
        cnn_out = layer1_input
    
    # Now transform this into a sent embedding
    sent_emb = tensor.dot(cnn_out,tparams['Wfc_sent']) + tparams['bfc_sent']
    
    return sent_emb, cnn_out, tparams, use_noise

# ========================================================================================
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

       # assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        self.max_pool_method = 'downsamp'
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape), 
                                                dtype=theano.config.floatX),name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),name="W_conv")   
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name="b_conv")
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = tensor.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        else:
            pooled_out = myMaxPool(conv_out, ps=self.poolsize, method=self.max_pool_method)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = {}
        self.params['CNN_W_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.W
        self.params['CNN_b_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.b

        
    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = Tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        else:
            pooled_out = myMaxPool(conv_out, ps=self.poolsize, method=self.max_pool_method)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
  
