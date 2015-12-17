import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split_theano
#from numbapro import cuda
from imagernn.lstm_generatorTheano import LSTMGenerator
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict

def main(params):
  batch_size = params['batch_size']
  word_count_threshold = params['word_count_threshold']
  max_epochs = params['max_epochs']
  host = socket.gethostname() # get computer hostname

  # fetch the data provider
  dp = getDataProvider(params)
  # Initialize the optimizer 
  solver = Solver(params['solver'])

  params['aux_inp_size'] = dp.aux_inp_size
  params['image_feat_size'] = dp.img_feat_size

  print 'Image feature size is %d, and aux input size is %d'%(params['image_feat_size'],params['aux_inp_size'])

  misc = {} # stores various misc items that need to be passed around the framework

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  if params['class_out_factoring'] == 0:
    misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)
  else:
    [misc['wordtoix'], misc['classes']], [misc['ixtoword'],  misc['clstotree'], misc['ixtoclsinfo']], [bias_init_vector,
        bias_init_inter_class] = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold, params)
    params['nClasses'] = bias_init_inter_class.shape[0]
    
  params['vocabulary_size'] = len(misc['wordtoix'])
  params['output_size'] = len(misc['ixtoword']) # these should match though
  print len(misc['wordtoix']),len(misc['ixtoword']) 

  # This initializes the model parameters and does matrix initializations  
  lstmGenerator = LSTMGenerator(params)
  model, misc['update'], misc['regularize'] = (lstmGenerator.model_th, lstmGenerator.update_list, lstmGenerator.regularize)
  
  # force overwrite here. The bias to the softmax is initialized to reflect word frequencies
  # This is a bit of a hack, not happy about it
  model['bd'].set_value(bias_init_vector.astype(config.floatX))
  if params['class_out_factoring'] == 1:
    model['bdCls'].set_value(bias_init_inter_class.astype(config.floatX))

  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion. 
  (use_dropout, inp_list,
     f_pred_prob, cost, predTh, updatesLSTM) = lstmGenerator.build_model(model, params)

  costGrad = cost[0]
  # Add class uncertainity to final cost
  #if params['class_out_factoring'] == 1:
  #  costGrad += cost[2]
  # Add the regularization cost. Since this is specific to trainig and doesn't get included when we 
  # evaluate the cost on test or validation data, we leave it here outside the model definition
  if params['regc'] > 0.:
      reg_cost = theano.shared(numpy_floatX(0.), name='reg_c')
      reg_c = tensor.as_tensor_variable(numpy_floatX(params['regc']), name='reg_c')
      reg_cost = 0.
      for p in misc['regularize']:
        reg_cost += (model[p] ** 2).sum()
        reg_cost *= 0.5 * reg_c 
      costGrad += (reg_cost /params['batch_size'])
    
  # Compile an evaluation function.. Doesn't include gradients
  # To be used for validation set evaluation
  f_eval= theano.function(inp_list, cost, name='f_eval')


  # Now let's build a gradient computation graph and rmsprop update mechanism
  grads = tensor.grad(costGrad, wrt=model.values())
  lr = tensor.scalar(name='lr',dtype=config.floatX)
  f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, grads,
                                      inp_list, cost, params)

  print 'model init done.'
  print 'model has keys: ' + ', '.join(model.keys())
  #print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['update'])
  #print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['regularize'])
  #print 'number of learnable parameters total: %d' % (sum(model[k].shape[0] * model[k].shape[1] for k in misc['update']), )

  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  num_sentences_total = dp.getSplitSize('train', ofwhat = 'sentences')
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  eval_period_in_epochs = params['eval_period']
  eval_period_in_iters = max(1, int(num_iters_one_epoch * eval_period_in_epochs))
  top_val_ppl2 = -1
  smooth_train_ppl2 = len(misc['ixtoword']) # initially size of dictionary of confusion
  val_ppl2 = len(misc['ixtoword'])
  last_status_write_time = 0 # for writing worker job status reports
  json_worker_status = {}
  json_worker_status['params'] = params
  json_worker_status['history'] = []

  len_hist = defaultdict(int)
  
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  if params['checkpoint_file_name'] != 'None':
    zipp(model_init_from,model)
    zipp(rg_init,rg)
    print("\nContinuing training from previous model\n. Already run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint_init['epoch'], \
      checkpoint_init['perplexity']))
  
  for it in xrange(max_iters):
    t0 = time.time()
    # fetch a batch of data
    if params['sample_by_len'] == 0:
        batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
    else: 
        batch,l = dp.getRandBatchByLen(batch_size)
        len_hist[l] += 1

    if params['use_pos_tag'] != 'None':
        real_inp_list, lenS = prepare_data(batch,misc['wordtoix'],params['maxlen'],sentTagMap,misc['ixtoword'],rev_sents=params['reverse_sentence'])
    else:    
        real_inp_list, lenS = prepare_data(batch,misc['wordtoix'],params['maxlen'], rev_sents=params['reverse_sentence'])
    
    # Enable using dropout in training 
    use_dropout.set_value(float(params['use_dropout']))
    epoch = it * 1.0 / num_iters_one_epoch
    
    if params['sched_sampling_mode'] !=None:
        real_inp_list.append(epoch)

    # evaluate cost, gradient and perform parameter update
    cost = f_grad_shared(*real_inp_list)
    f_update(params['learning_rate'])
    dt = time.time() - t0

    # print training statistics
    train_ppl2 = (2**(cost[1]/lenS)) #step_struct['stats']['ppl2']
    smooth_train_ppl2 = 0.99 * smooth_train_ppl2 + 0.01 * train_ppl2 # smooth exponentially decaying moving average
    if it == 0: smooth_train_ppl2 = train_ppl2 # start out where we start out
    total_cost = cost[0]
    #print '%d/%d batch done in %.3fs. at epoch %.2f. loss cost = %f, reg cost = %f, ppl2 = %.2f (smooth %.2f)' \
    #      % (it, max_iters, dt, epoch, cost['loss_cost'], cost['reg_cost'], \
    #         train_ppl2, smooth_train_ppl2)

    tnow = time.time()
    if tnow > last_status_write_time + 60*1: # every now and then lets write a report
      print '%d/%d batch done in %.3fs. at epoch %.2f. Cost now is %.3f and pplx is %.3f' % (it, max_iters, dt, \
		    epoch, total_cost, train_ppl2)
      last_status_write_time = tnow
      jstatus = {}
      jstatus['time'] = datetime.datetime.now().isoformat()
      jstatus['iter'] = (it, max_iters)
      jstatus['epoch'] = (epoch, max_epochs)
      jstatus['time_per_batch'] = dt
      jstatus['smooth_train_ppl2'] = smooth_train_ppl2
      jstatus['val_ppl2'] = val_ppl2 # just write the last available one
      jstatus['train_ppl2'] = train_ppl2
      #if params['class_out_factoring'] == 1:
      #  jstatus['class_cost'] = float(cost[2])
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
          top_val_ppl2 = val_ppl2
          filename = 'model_checkpoint_%s_%s_%s_%.2f.p' % (params['dataset'], host, params['fappend'], val_ppl2)
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
          checkpoint['misc'] = misc
          try:
            pickle.dump(checkpoint, open(filepath, "wb"))
            print 'saved checkpoint in %s' % (filepath, )
          except Exception, e: # todo be more clever here
            print 'tried to write checkpoint into %s but got error: ' % (filepath, )
            print e

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('--use_theano', dest='use_theano', default=1, help='Should we use thano and gpu!?. Actually dont try with value 0 :-|')

  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--worker_status_output_directory', dest='worker_status_output_directory', type=str, default='status/', help='directory to write worker status JSON blobs to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--continue_training', dest='checkpoint_file_name', type=str, default='None', help='checkpoint file from which to resume training')
  parser.add_argument('--use_pos_tag', dest='use_pos_tag', type=str, default='None', help='use_pos_tag')

  # Some parameters about image features used
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='vgg_feats.mat', help='Which file should we use for read the CNN features')
  parser.add_argument('--image_feat_size', dest='image_feat_size', type=int, default=4096, help='size of the input image features')
  parser.add_argument('--data_file', dest='data_file', type=str, default='dataset.json', help='Which dataset file shpuld we use')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=-1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default='None', help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--swap_AuxFeat', dest='swap_aux', type=int, default=0, help='Feed image features through auxillary input!')
 
  # Parameters to enable class based factorization of lstm output
  parser.add_argument('--class_out_factoring', dest='class_out_factoring', type=int, default=0, help='Enable Class based output factorization in generator')
  parser.add_argument('--nClasses', dest='nClasses', type=int, default=200, help='Number of classes to use')
  parser.add_argument('--class_inp_file', dest='class_inp_file', type=str, default=None, help='If clustering is already done, provide the inp file')
  parser.add_argument('--clust_tool_dir', dest='clust_tool_dir', type=str, default=None, help='Needed if clustering needs to be triggered now.' 
                                                                        ' Support currently is only for Percy Liang\'s Brown clustering implementation')

  # model parameters
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=512, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use')
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, help='regularization strength')
  parser.add_argument('--tanhC_version', dest='tanhC_version', type=int, default=0, help='use tanh version of LSTM?')

  # optimization parameters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver types supported: rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1, help='enable or disable dropout')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')
  parser.add_argument('--drop_prob_aux', dest='drop_prob_aux', type=float, default=0.5, help='what dropout to apply for the auxillary inputs to lstm')
  
  # refere paper: "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" http://arxiv.org/abs/1506.03099
  parser.add_argument('--sched_sampling_mode', dest='sched_sampling_mode', type=str, default=None, help='should we implement scheduled sampling during training')
  parser.add_argument('--sched_sampling_const', dest='sched_sampling_const', type=float, default=1.0, help='scheduling constant, exact nature depends on the mode')
  parser.add_argument('--sslin_slope', dest='sslin_slope', type=np.float, default=1.0, help='slope of decay in linear scheduling')
  parser.add_argument('--sslin_min', dest='sslin_min', type=np.float, default=1.0, help='min truth constant in linear scheduling')
  
  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=0, help='enable sampling by length of sentece to speed up training')
  parser.add_argument('--maxlen', dest='maxlen', type=int, default=None, help='enable sampling by length of sentece to speed up training')
  
  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=1.0, help='decay factor for learning rate, applied every epoch')
  parser.add_argument('--lr_decay_st_epoch', dest='lr_decay_st_epoch', type=float, default=100.0, help='from which epoch should the lr decay start')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')
  parser.add_argument('--reverse_sentence', dest='reverse_sentence', type=int, default=0, help='Should we reverse the sentences when feeding it to the RNN?')
  parser.add_argument('--use_video_feat', dest='use_video_feat', type=int, default=0, help='Use video features for training')

  # evaluation parameters
  parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, default=1.0, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=100, help='for faster validation performance evaluation, what batch size to use on val img/sentences?')
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
  
  # parameters controlling use of external data (For eg. lsmdc uses COCO Nearest neighbhors)
  parser.add_argument('--ext_data_file', dest='ext_data_file', type=str, default=None, help='Which external dataset file shpuld we use')
  parser.add_argument('--ed_sample_prob', dest='ed_sample_prob', type=np.float, default=0.0, help='probability with which to sample external data')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['checkpoint_file_name'] != 'None':
    checkpoint_init = pickle.load(open(params['checkpoint_file_name'], 'rb'))
    model_init_from = checkpoint_init['model']
    rg_init = checkpoint_init.get('rgrads',[])

  if params['aux_inp_file'] != 'None':
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0
  
  if params['use_pos_tag'] != 'None':
    sentTagMap = pickle.load(open(params['use_pos_tag'],'r'))  
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.allow_gc = False
  config.exception_verbosity = 'high'
  main(params)
