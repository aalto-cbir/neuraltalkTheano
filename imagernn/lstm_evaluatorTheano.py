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
from copy import copy

class LSTMEvaluator:
  """ 
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params, Wemb=None):

    word_encoding_size = params.get('word_encoding_size', 128)
    aux_inp_size = params.get('aux_inp_size', -1)
    sent_encoding_size = params.get('sent_encoding_size',-1)# size of CNN vectors hardcoded here
    self.sent_enc_size = sent_encoding_size
    
    # Output state is the sentence encoding
    output_size = sent_encoding_size
    
    hidden_size = params.get('hidden_size', 128)
    hidden_depth = params.get('hidden_depth', 1)
    generator = params.get('generator', 'lstm')
    vocabulary_size = params.get('vocabulary_size',-1)
    image_feat_size = params.get('image_feat_size',-1)# size of CNN vectors hardcoded here

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    img_encoding_size = self.sent_enc_size if(params.get('multimodal_lstm',0) == 0) else word_encoding_size
    if params.get('swap_aux',0) == 0:
        model['WIemb'] = initwTh(image_feat_size, img_encoding_size) # image encoder
    else:
        model['WIemb'] = initwTh(aux_inp_size, image_encoding_size) # image encoder
    model['b_Img'] = np.zeros((img_encoding_size)).astype(config.floatX)
    
    if Wemb == None:
        model['Wemb'] = initwTh(vocabulary_size, word_encoding_size) # word encoder
    
    model['lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model['lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)

    for i in xrange(1,hidden_depth):
        model['lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)

    model['lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)
    if(params.get('multimodal_lstm',0) == 0):
        model['Wd'] = initwTh(hidden_size, output_size) # decoder
        model['bd'] = np.zeros((output_size,)).astype(config.floatX)

    update_list = ['lstm_W_hid', 'lstm_W_inp', 'lstm_b', 'Wd', 'bd', 'WIemb', 'b_Img', 'Wemb']
    self.regularize = ['lstm_W_hid', 'lstm_W_inp', 'Wd', 'WIemb', 'Wemb' ]
    
    for i in xrange(1,hidden_depth):
        update_list.append('lstm_W_hid_'+str(i))
        update_list.append('lstm_W_hid_'+str(i))
        self.regularize.append('lstm_W_inp_'+str(i))
        self.regularize.append('lstm_W_inp_'+str(i))

    self.model_th = self.init_tparams(model)
    self.updateP = update_list 

# ========================================================================================
  def init_tparams(self,params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# ========================================================================================
 # BUILD LSTM forward propogation model
  def build_model(self, tparams, optionsInp):
    trng = RandomStreams(1234)
    options = copy(optionsInp)
    if 'en_aux_inp' in options:
        options.pop('en_aux_inp')
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    xW = tensor.matrix('xW', dtype='int64')
    mask = tensor.vector('mask', dtype='int64')
    
    n_Rwords= xW.shape[0]
    n_samples = xW.shape[1]

    embW = tparams['Wemb'][xW.flatten()].reshape([n_Rwords,
                                                n_samples,
                                                options['word_encoding_size']])
    xI = tensor.matrix('xI', dtype=config.floatX)
    
    if options.get('multimodal_lstm',0) == 1:
        embImg = tensor.dot(xI, tparams['WIemb']) + tparams['b_Img']
        embImg = tensor.shape_padleft(tensor.extra_ops.repeat(embImg,n_samples,axis=0),n_ones=1)
        emb = tensor.concatenate([embImg, embW], axis=0)
    else:
        emb = embW

    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, options['drop_prob_encoder'], shp = emb.shape)

    # This implements core lstm
    rval, updatesLSTM = basic_lstm_layer(tparams, emb, [], use_noise, options, prefix='lstm')

    if options['use_dropout']:
        p = dropout_layer(sliceT(rval[0][mask + options.get('multimodal_lstm',0),tensor.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,
            options['hidden_size']), use_noise, trng, options['drop_prob_decoder'], (n_samples,options['hidden_size']))
    else:
        p = sliceT(rval[0][mask + options.get('multimodal_lstm',0),tensor.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,options['hidden_size'])

    
    if options.get('multimodal_lstm',0) == 0:
        sent_emb = (tensor.dot(p,tparams['Wd']) + tparams['bd'])
        probMatch, sim_score = multimodal_cosine_sim_softmax(xI, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    else:
        sent_emb = tensor.sum(p,axis=1).T #(tensor.dot(p,tparams['Wd'])).T
        sim_score = sent_emb #tensor.maximum(0.0, sent_emb) #tensor.tanh(sent_emb) 
        smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('sim_smooth_factor',1.0)), name='sm_f')
        probMatch = tensor.nnet.softmax(sim_score*smooth_factor)
        
    inp_list = [xW, mask, xI]
    
    if options.get('mode','batchtrain') == 'batchtrain':
        # In train mode we compare a batch of images against each others captions.
        batch_size = options['batch_size']
        cost = -(tensor.log(probMatch.diagonal()).sum())/batch_size
    else:
        # In predict mode we compare multiple captions against a single image 
        posSamp = tensor.ivector('posSamp')
        batch_size = posSamp.shape[0] 
        cost = -(tensor.log(probMatch[0,posSamp]).sum())/batch_size
        inp_list.append(posSamp)
    
    f_pred_sim_prob = theano.function(inp_list[:3], probMatch, name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(inp_list[:3], sim_score, name='f_pred_sim_scr')
    if options.get('multimodal_lstm',0) == 1:
        f_sent_emb = theano.function([inp_list[0],inp_list[2]], [rval[0],emb], name='f_sent_emb')
    else:
        f_sent_emb = theano.function([inp_list[0]], [rval[0],emb], name='f_sent_emb')

    return use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb, updatesLSTM], cost, sim_score, tparams 

# ========================================================================================
 # BUILD LSTM forward propogation eval model with ability to take direct Wemb inputs from gen model
  def build_advers_eval(self, tparams, optionsInp, gen_inp_list, gen_out):
    trng = RandomStreams(1234)
    options = copy(optionsInp)
    options.pop('en_aux_inp')
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    xW = tensor.matrix('xW', dtype='int64')
    maskRef = tensor.vector('maskEval', dtype='int64')
    
    n_Rwords = xW.shape[0]
    n_samples = xW.shape[1]
    n_gen_words = gen_out.shape[0]
    gen_emb = gen_out.dot(tparams['Wemb'])

    embWRef = tparams['Wemb'][xW.flatten()].reshape([n_Rwords,
                                                n_samples,
                                                options['word_encoding_size']])
    embGen = ifelse(n_Rwords > n_gen_words,tensor.concatenate([gen_emb,theano.tensor.alloc(numpy_floatX(0.), 
                                                       n_Rwords-n_gen_words,options['word_encoding_size'])], axis=0),
                                                       gen_emb[:n_Rwords,:])
    embGen = embGen.dimshuffle(0,'x',1)
    
    xImg = gen_inp_list[options['swap_aux']]

    emb = tensor.concatenate([embWRef, embGen], axis=1)
    n_samples = n_samples + 1
    mask = tensor.concatenate([maskRef, tensor.minimum(n_Rwords-1,n_gen_words-1).dimshuffle('x')]) 
    

    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, options['drop_prob_encoder'], shp = emb.shape)

    # This implements core lstm
    rval, updatesLSTM = basic_lstm_layer(tparams, emb, [], use_noise, options, prefix='lstm')

    if options['use_dropout']:
        p = dropout_layer(sliceT(rval[0][mask,tensor.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,options['hidden_size']), use_noise, trng,
            options['drop_prob_decoder'], (n_samples,options['hidden_size']))
    else:
        p = sliceT(rval[0][mask,tensor.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,options['hidden_size'])

    sent_emb = tensor.dot(p,tparams['Wd']) + tparams['bd']

    #if options['rand_negs'] > 0:
    #    good_ref_sents = tensor.arange(n_samples-1-options['rand_negs'])

    
    if options.get('sim_metric','cosine') == 'cosine':
        print 'Using cosine Similarity'
        probMatch, sim_score = multimodal_cosine_sim_softmax(xImg, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    else:    
        print 'Using Eucledean distance'
        probMatch, sim_score = multimodal_euc_dist_softmax(xImg, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
        
    inp_list = [xW, maskRef]
    for inp in gen_inp_list:
      if inp not in inp_list:
          inp_list.append(inp)
    
    smooth_eps = tensor.constant(options['smooth_eps'],dtype=config.floatX)
    costEval = -tensor.log(probMatch[0,0]+smooth_eps)# + tensor.log(probMatch[0,-1]+smooth_eps)
    costGen = -tensor.log(probMatch[0,-1]+smooth_eps)
    
    f_pred_sim_prob = theano.function(inp_list, probMatch, name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(inp_list, sim_score, name='f_pred_sim_scr')
    f_sent_emb = theano.function(inp_list, sent_emb, name='f_sent_emb')

    return use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb, updatesLSTM], [costEval, costGen], sim_score, tparams 
