import argparse
import json
import time
import datetime
import numpy as np
import random
import code
import socket
import os
import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator 
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict

def eval_split_theano(split, dp, model, params, misc, gen_fprop, **kwargs):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  eval_batch_size = kwargs.get('batch_size', params.get('batch_size',50))
  eval_max_images = kwargs.get('max_images', params.get('max_images', -1))
  wordtoix = misc['wordtoix']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  avg_scr= 0
  logppln = 0
  nsent = 0
  if params['mode'] == 'batchtrain' :
    pos_samp = np.arange(eval_batch_size,dtype=np.int32)
    for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images,shuffle = True):
      if len(batch) < eval_batch_size:
          break;
      inp_list, lenS = prepare_data(batch,wordtoix,maxlen=params['maxlen'],pos_samp=pos_samp,prep_for=params['eval_model'])
      scrs = gen_fprop(*inp_list)
      avg_scr += scrs
      logppln += lenS 
      nsent += 1 
  else:
    pos_samp = np.arange(1,dtype=np.int32)
    for img in dp.iterImages(split = split, max_images = eval_max_images,shuffle = True):
        batch = []
        if params['mode'] == 'finetune':
            for si in img['prefOrder'][:eval_batch_size]:
                batch.append({'sentence':img['sentences'][si]})
            # To keep all the batches of same size, pad if necessary
            for i in xrange(eval_batch_size - len(img['prefOrder'])):
                batch.append({'sentence':img['sentences'][-1]})
        else:
            img_idx = np.random.choice(np.arange(len(dp.split[split])), size=eval_batch_size-1, replace=False)
            batch.append({'sentence': random.choice(img['sentences'])})
            for i in img_idx:
                batch.append({'sentence':random.choice(dp.split['train'][i]['sentences'])})
        # Finally store image feature
        batch[0]['image'] = img
        inp_list, lenS = prepare_data(batch,wordtoix,maxlen=params['maxlen'],pos_samp=pos_samp,prep_for=params['eval_model'])
        inp_list.append(pos_samp)
        scrs = gen_fprop(*inp_list)
        avg_scr += np.e**-scrs
        logppln += lenS 
        nsent += 1

  average_score = avg_scr/ nsent 
  print 'evaluated %d sentences and got perplexity = %f' % (nsent, average_score)
  return average_score# return the perplexity

#######################################################################################################
def main(params):
  word_count_threshold = params['word_count_threshold']
  max_epochs = params['max_epochs']
  host = socket.gethostname() # get computer hostname

  # fetch the data provider
  dp = getDataProvider(params)
  # Initialize the optimizer 
  solver = Solver(params['solver'])

  params['image_feat_size'] = dp.img_feat_size

  misc = {} # stores various misc items that need to be passed around the framework

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)
  params['use_dropout'] = 1 

  if params['fine_tune'] == 1:
    params['mode'] = 'multimodal_lstm' if params['multimodal_lstm'] == 0 else 'multimodal_lstm'
    if params['checkpoint_file_name'] != None:
        params['batch_size'] = dp.dataset['batchsize']
        misc['wordtoix'] = checkpoint_init['wordtoix']
        misc['ixtoword'] = checkpoint_init['ixtoword']
    batch_size = 1
    num_sentences_total = dp.getSplitSize('train', ofwhat = 'images')
  else:
    params['mode'] = 'batchtrain'
    batch_size = params['batch_size']
    num_sentences_total = dp.getSplitSize('train', ofwhat = 'sentences')
  
  params['vocabulary_size'] = len(misc['wordtoix'])
  pos_samp = np.arange(batch_size,dtype=np.int32)

  # This initializes the model parameters and does matrix initializations  
  evalModel = decodeEvaluator(params)
  model, misc['update'], misc['regularize'] = (evalModel.model_th, evalModel.updateP, evalModel.regularize)
  
  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion. 
  (use_dropout, inp_list,
     miscOuts, cost, predTh, model) = evalModel.build_model(model, params)

  # Add the regularization cost. Since this is specific to trainig and doesn't get included when we 
  # evaluate the cost on test or validation data, we leave it here outside the model definition
  if params['regc'] > 0.:
      reg_cost = theano.shared(numpy_floatX(0.), name='reg_c')
      reg_c = tensor.as_tensor_variable(numpy_floatX(params['regc']), name='reg_c')
      reg_cost = 0.
      for p in misc['regularize']:
        reg_cost += (model[p] ** 2).sum()
        reg_cost *= 0.5 * reg_c 
      cost[0] += (reg_cost /params['batch_size'])
    
  # Compile an evaluation function.. Doesn't include gradients
  # To be used for validation set evaluation
  f_eval= theano.function(inp_list, cost, name='f_eval')

  # Now let's build a gradient computation graph and rmsprop update mechanism
  grads = tensor.grad(cost, wrt=model.values())
  lr = tensor.scalar(name='lr',dtype=config.floatX)
  if params['sim_minibatch'] > 0:
    f_grad_accum, f_clr, ag = solver.accumGrads(model,grads,inp_list,cost, params['sim_minibatch'])
    f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, ag,
                                      inp_list, cost, params)
  else: 
    f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, grads,
                                      inp_list, cost, params)

  print 'model init done.'
  print 'model has keys: ' + ', '.join(model.keys())

  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  inner_loop =   params['sim_minibatch'] if params['sim_minibatch'] > 0 else 1
  max_iters = max_iters / inner_loop 
  eval_period_in_epochs = params['eval_period']
  eval_period_in_iters = max(1, int(num_iters_one_epoch * eval_period_in_epochs/ inner_loop))
  top_val_ppl2 = -1
  smooth_train_cost = len(misc['ixtoword']) # initially size of dictionary of confusion
  val_ppl2 = len(misc['ixtoword'])
  last_status_write_time = 0 # for writing worker job status reports
  json_worker_status = {}
  json_worker_status['params'] = params
  json_worker_status['history'] = []

  len_hist = defaultdict(int)
  
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  if params['checkpoint_file_name'] != None:
    zipp(model_init_from,model)
    zipp(rg_init,rg)
    print("\nContinuing training from previous model\n. Already run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint_init['epoch'], \
      checkpoint_init['perplexity']))
  elif params['init_from_imagernn'] != None:
    # Initialize word vecs and image emb from generative model file
    rnnCv = pickle.load(open(params['init_from_imagernn'], 'rb'))
    model['Wemb'].set_value(rnnCv['model']['Wemb'])
    model['WIemb'].set_value(rnnCv['model']['WIemb_aux'])
    misc['wordtoix'] = rnnCv['wordtoix']
    misc['ixtoword'] = rnnCv['ixtoword']
    print("\n Initialized Word embedding and Image embeddings from gen mode %s" % (params['init_from_imagernn']))


  use_dropout.set_value(1.)
  #################### Main Loop ############################################
  for it in xrange(max_iters):
    t0 = time.time()
    # fetch a batch of data
    cost_inner = np.zeros((inner_loop,),dtype=np.float32)
    if params['sim_minibatch'] > 0:
        for i_l in xrange(inner_loop):
            batch,pos_samp_sent = dp.sampPosNegSentSamps(params['batch_size'],params['mode'],thresh=0.3) 
            real_inp_list, lenS = prepare_data(batch,misc['wordtoix'],maxlen=params['maxlen'],pos_samp=pos_samp,prep_for=params['eval_model'])
            if params['fine_tune'] == 1:
               real_inp_list.append(pos_samp_sent)
            cost_inner[i_l] = f_grad_accum(*real_inp_list)
    else:
        batch,pos_samp_sent = dp.sampPosNegSentSamps(params['batch_size'],params['mode'],thresh=0.3)
        real_inp_list, lenS = prepare_data(batch,misc['wordtoix'],maxlen=params['maxlen'],pos_samp=pos_samp,prep_for=params['eval_model'])
        if params['fine_tune'] == 1:
           real_inp_list.append(pos_samp_sent)
    # Enable using dropout in training 
    cost = f_grad_shared(*real_inp_list)
    f_update(params['learning_rate'])
    dt = time.time() - t0
   
    # Reset accumulated gradients to 0
    if params['sim_minibatch'] > 0:
        f_clr()
    #print 'model: ' + ' '.join([str(np.isnan(model[m].get_value()).any()) for m in model])
    #print 'rg: ' +' '.join([str(np.isnan(rg[i].get_value()).any()) for i in xrange(len(rg))])
    #print 'zg: ' + ' '.join([str(np.isnan(zg[i].get_value()).any()) for i in xrange(len(zg))])
    #print 'ud: ' + ' '.join([str(np.isnan(ud[i].get_value()).any()) for i in xrange(len(ud))])
    #import pdb; pdb.set_trace()
    #print 'udAft: ' + ' '.join([str(np.isnan(ud[i].get_value()).any()) for i in xrange(len(ud))])

    # print training statistics
    epoch = it*inner_loop * 1.0 / num_iters_one_epoch
    total_cost = (np.e**-cost + (np.e**(-cost_inner)).sum()*(params['sim_minibatch'] > 0))/ (1 + params['sim_minibatch'])
    #print '%d/%d batch done in %.3fs. at epoch %.2f. loss cost = %f, reg cost = %f, ppl2 = %.2f (smooth %.2f)' \
    #      % (it, max_iters, dt, epoch, cost['loss_cost'], cost['reg_cost'], \
    #         train_ppl2, smooth_train_cost)
    if it == 0: smooth_train_cost = total_cost 
    else: smooth_train_cost = 0.99 * smooth_train_cost + 0.01 * total_cost

    tnow = time.time()
    if tnow > last_status_write_time + 60*1: # every now and then lets write a report
      print '%d/%d batch done in %.3fs. at epoch %.2f. Prob now is %.3f' % (it, max_iters, dt, \
		epoch, smooth_train_cost)
      last_status_write_time = tnow
      jstatus = {}
      jstatus['time'] = datetime.datetime.now().isoformat()
      jstatus['iter'] = (it, max_iters)
      jstatus['epoch'] = (epoch, max_epochs)
      jstatus['time_per_batch'] = dt
      jstatus['val_ppl2'] = val_ppl2 # just write the last available one
      json_worker_status['history'].append(jstatus)
      status_file = os.path.join(params['worker_status_output_directory'], host + '_status.json')
      #import pdb; pdb.set_trace()
      try:
        json.dump(json_worker_status, open(status_file, 'w'))
      except Exception, e: # todo be more clever here
        print 'tried to write worker status into %s but got error:' % (status_file, )
        print e
    
    ## perform perplexity evaluation on the validation set and save a model checkpoint if it's good
    is_last_iter = (it+1) == max_iters
    if (((it+1) % eval_period_in_iters) == 0 and it < max_iters - 5) or is_last_iter:
      # Disable using dropout in validation 
      use_dropout.set_value(0.)

      val_ppl2 = eval_split_theano('val', dp, model, params, misc,f_eval) # perform the evaluation on VAL set
      if epoch - params['lr_decay_st_epoch'] >= 0:
        params['learning_rate'] = params['learning_rate'] * params['lr_decay']
        params['lr_decay_st_epoch'] += 1
      
      print 'validation perplexity = %f, lr = %f' % (val_ppl2, params['learning_rate'])
      if params['sample_by_len'] == 1:
        print len_hist

      write_checkpoint_ppl_threshold = params['write_checkpoint_ppl_threshold']
      if val_ppl2 < top_val_ppl2 or top_val_ppl2 < 0:
        if val_ppl2 < write_checkpoint_ppl_threshold or write_checkpoint_ppl_threshold < 0:
          # if we beat a previous record or if this is the first time
          # AND we also beat the user-defined threshold or it doesnt exist
          #top_val_ppl2 = val_ppl2
          filename = '%s_checkpoint_%s_%s_%s_%.2f_%.2f.p' % (params['eval_model'], params['dataset'], host, params['fappend'],val_ppl2,smooth_train_cost)
          filepath = os.path.join(params['checkpoint_output_directory'], filename)
          model_npy = unzip(model)
          rgrads_npy = unzip(rg)
          checkpoint = {}
          checkpoint['it'] = it
          checkpoint['epoch'] = epoch
          checkpoint['model'] = model_npy
          checkpoint['rgrads'] = rgrads_npy
          checkpoint['params'] = params
          checkpoint['perplexity'] = val_ppl2
          checkpoint['wordtoix'] = misc['wordtoix']
          checkpoint['ixtoword'] = misc['ixtoword']
          try:
            pickle.dump(checkpoint, open(filepath, "wb"))
            print 'saved checkpoint in %s' % (filepath, )
          except Exception, e: # todo be more clever here
            print 'tried to write checkpoint into %s but got error: ' % (filepath, )
            print e

      use_dropout.set_value(1.)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('--use_theano', dest='use_theano', default=1, help='Should we use thano and gpu!?. Actually dont try with value 0 :-|')

  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: flickr8k/flickr30k')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cnnCv/', help='output directory to write checkpoints to')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('--worker_status_output_directory', dest='worker_status_output_directory', type=str, default='status/', help='directory to write worker status JSON blobs to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--continue_training', dest='checkpoint_file_name', type=str, default=None, help='checkpoint file from which to resume training')
  parser.add_argument('--init_from_imagernn', dest='init_from_imagernn', type=str, default=None, help='Gen model cv to initialize word vecs and image emb from')

  # Some parameters about image features used
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='vgg_feats.mat', help='Which file should we use for read the CNN features')
  parser.add_argument('--image_feat_size', dest='image_feat_size', type=int, default=4096, help='size of the input image features')
  parser.add_argument('--data_file', dest='data_file', type=str, default='dataset.json', help='Which dataset file shpuld we use')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=-1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  parser.add_argument('--fine_tune', dest='fine_tune', type=int, default=0, help='whether to run on one img at at time or batch_size images')
  
  # model parameters
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--sent_encoding_size', dest='sent_encoding_size', type=int, default=512, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--max_sent_len', dest='maxlen', type=int, default=15, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--sim_smooth_factor', dest='sim_smooth_factor', type=float, default=1.0, help='smoothing factor in softmax')
  parser.add_argument('--eval_model', dest='eval_model', type=str, default='cnn', help='which evaluator model to use type: cnn/lstm_eval')
  parser.add_argument('--multimodal_lstm', dest='multimodal_lstm', type=int, default=0, help='If 1, we will feed the image encoding at t=0 to lstm and no other similarity metric is used')
 
  # CNN specific parameters
  parser.add_argument('--n_fmaps', dest='n_fmaps_psz', type=int, default=100, help='number of cnn feature maps per filter height')
  parser.add_argument('--filter_hs', dest='filter_hs', metavar='N', type=int, nargs='+',default =[3,4,5], help='list fo filter heights to use in CNN')
  parser.add_argument('--conv_non_linear', dest='conv_non_linear', type=str, default='relu', help='nonlinearity type: tanh/relu')

  # LSTM Specific parameters
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')

  # optimization parameters
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=0., help='regularization strength')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=128, help='batch size')
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver type: vanilla/adagrad/adadelta/rmsprop')
  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=1.0, help='decay factor for learning rate, applied every epoch')
  parser.add_argument('--lr_decay_st_epoch', dest='lr_decay_st_epoch', type=float, default=100.0, help='from which epoch should the lr decay start')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.0, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=0, help='enable sampling by length of sentece to speed up training')
  parser.add_argument('--sim_minibatch', dest='sim_minibatch', type=int, default=0, help='If >0, we will accumulate grads for this many iters before applying update')
  
  # Droput Regularization related
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=np.float32, default=0.5, help='what dropout to apply right befor the decoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=np.float32, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_cnn', dest='drop_prob_cnn', type=np.float32, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')
  parser.add_argument('--drop_prob_aux', dest='drop_prob_aux', type=np.float32, default=0.5, help='what dropout to apply for the auxillary inputs to lstm')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')

  # evaluation parameters
  parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, default=1.0, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=100, help='for faster validation performance evaluation, what batch size to use on val img/sentences?')
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['checkpoint_file_name'] != None:
    checkpoint_init = pickle.load(open(params['checkpoint_file_name'], 'rb'))
    model_init_from = checkpoint_init['model']
    rg_init = checkpoint_init.get('rgrads',[])

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.profile = True
  #config.allow_gc = False
  main(params)
