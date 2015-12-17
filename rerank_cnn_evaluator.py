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
from imagernn.data_provider import getDataProvider, prepare_data, loadArbitraryFeatures
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator 
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict


#######################################################################################################
def main(params):
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  model_npy = checkpoint['model']
  
  # Load the candidates db generated from rnn's
  candDb = json.load(open(params['candDb'],'r'))
  wordtoix = checkpoint['wordtoix']

  #find the number of candidates per image and max sentence len
  batch_size = 0
  maxlen = 0
  for i,img in enumerate(candDb['imgblobs']):
    for ids,cand in enumerate(img['candidatelist']):
        tks = cand['text'].split(' ')
        # Also tokenize the candidates
        candDb['imgblobs'][i]['candidatelist'][ids]['tokens'] = tks
        if len(tks) > maxlen:
            maxlen = len(tks)
    if batch_size < len(img['candidatelist']):
        batch_size = len(img['candidatelist'])

  # Get all images to this batch size!
  # HACK!!
  maxlen = 24
  checkpoint_params['maxlen'] = maxlen
 
  checkpoint_params['batch_size'] = batch_size
  print maxlen

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  
  # This initializes the model parameters and does matrix initializations  
  checkpoint_params['mode'] = 'predict' 
  evalModel = decodeEvaluator(checkpoint_params)
  model = evalModel.model_th
  
  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion. 
  (use_dropout, inp_list,
     f_pred_fns, cost, predTh, model) = evalModel.build_model(model, checkpoint_params)

  # Add the regularization cost. Since this is specific to trainig and doesn't get included when we 
  # evaluate the cost on test or validation data, we leave it here outside the model definition

  # Now let's build a gradient computation graph and rmsprop update mechanism
  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  zipp(model_npy,model)
  print("\nPredicting using model %s, run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint_path, checkpoint['epoch'], \
    checkpoint['perplexity']))
  
  pos_samp = np.arange(1,dtype=np.int32)
  
  features,_ = loadArbitraryFeatures(params, -1)

  #Disable using dropout in training 
  use_dropout.set_value(0.)
  N = len(candDb['imgblobs'])
  #################### Main Loop ############################################
  for i,img in enumerate(candDb['imgblobs']):
    # fetch a batch of data
    print 'image %d/%d  \r' % (i, N),
    batch = []
    cbatch_len  = len(img['candidatelist'])
    for s in img['candidatelist']:
        batch.append({'sentence':s})
    if cbatch_len < batch_size:
        for z in xrange(batch_size - cbatch_len):
            batch.append({'sentence':img['candidatelist'][-1]})

    batch[0]['image'] = {'feat':features[:, img['imgid']]}
    real_inp_list, lenS = prepare_data(batch, wordtoix, maxlen=maxlen, pos_samp=pos_samp, prep_for=checkpoint_params['eval_model'])
    
    # evaluate cost, gradient and perform parameter update
    scrs = np.squeeze(f_pred_fns[1](*real_inp_list))
    scrs = scrs[:cbatch_len] # + scrs[:,cbatch_len:].sum()/cbatch_len
    for si,s in enumerate(img['candidatelist']):
        candDb['imgblobs'][i]['candidatelist'][si]['logprob'] = float(scrs[si])
        candDb['imgblobs'][i]['candidatelist'][si].pop('tokens')
    bestcand = scrs.argmax()
    candDb['imgblobs'][i]['candidate'] = candDb['imgblobs'][i]['candidatelist'][bestcand]
    srtidx = np.argsort(scrs)[::-1]
    candDb['imgblobs'][i]['candsort'] = list(srtidx)
    #import pdb;pdb.set_trace()
    # print training statistics

  print ""
  jsonFname = '%s_reranked_%s.json' % (checkpoint_params['eval_model'],params['fname_append'])
  save_file = os.path.join(params['root_path'], jsonFname)
  json.dump(candDb, open(save_file, 'w'))


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint of cnn evaluator')
  parser.add_argument('candDb', type=str, help='the candidate result file')
  parser.add_argument('-f', '--feat_file', type=str, default='vgg_feats.mat', help='file with the features. We can rightnow process only .mat format') 
  parser.add_argument('-d', '--dest', dest='root_path', default='example_images', type=str, help='folder to store the output files')
  parser.add_argument('--fname_append', type=str, default='', help='str to append to routput files')

  # Some parameters about image features used
  # model parameters
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.profile = True
  #config.allow_gc = False
  #main(params)
