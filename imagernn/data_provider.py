import json

import os
import h5py
import scipy.io
import codecs
import numpy as np
import theano
import random
from collections import defaultdict
from picsom_bin_data import picsom_bin_data

class BasicDataProvider:
  def __init__(self, params):
    dataset = params.get('dataset', 'coco')
    feature_file = params.get('feature_file', 'vgg_feats.mat')
    data_file = params.get('data_file', 'dataset.json')
    mat_new_ver = params.get('mat_new_ver', -1)
    print 'Initializing data provider for dataset %s...' % (dataset, )
    self.hdf5Flag = 0 #Flag indicating whether the dataset is an HDF5 File.
                 #Large HDF5 files are stored (by Vik) as one image 
                 #  per row, going against the conventions of the other
                 # storage formats

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')
    self.use_video_feat = params['use_video_feat']

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, data_file)
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))
    
    # load external data to augment the training captions
    self.use_extdata = 0
    if params.get('ext_data_file',None) != None:
        print 'BasicDataProvider: reading ext data file %s' % (params['ext_data_file'], )
        self.ext_data = json.load(open(params['ext_data_file'],'r'))
        self.use_extdata = 1
        self.edata_prob = params['ed_sample_prob']

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, feature_file)
    print 'BasicDataProvider: reading %s' % (features_path, )
    
    ## A hook to hack, no longer needed as dataset.json for lsmdc has been fixed
    offset = 0 if dataset == 'lsmdc2015' else 0
    print offset

    self.capperimage = self.dataset.get('capperimage',5)
    print self.capperimage
    
    if feature_file.rsplit('.',1)[1] == 'mat':
        if mat_new_ver == 1:
            features_struct = h5py.File(features_path)
            self.features = np.array(features_struct[features_struct.keys()[0]],dtype=theano.config.floatX)
        else:
            features_struct = scipy.io.loadmat(features_path)
            self.features = features_struct['feats']
    #The condition below makes consuming HDF5 features easy
    #   This is what I (Vik) use for features extracted in an unsupervised
    #   manner
    elif feature_file.rsplit('.',1)[1] == 'hdf5':
        self.hdf5Flag = 1
        features_struct = h5py.File(features_path)
        self.features = features_struct['features'] #The dataset in the HDF5 file is named 'features'
    elif feature_file.rsplit('.',1)[1] == 'bin':
        #This is for feature concatenation.
        # NOTE: Assuming same order of features in all the files listed in the txt file
        features_struct = picsom_bin_data(features_path) 
        self.features = np.array(features_struct.get_float_list(-1))[offset:,:].T.astype(theano.config.floatX) 
		# this is a 4096 x N numpy array of features
        print "Working on Bin file now"                                        
    elif feature_file.rsplit('.',1)[1] == 'txt':
        #This is for feature concatenation.
        # NOTE: Assuming same order of features in all the files listed in the txt file
        feat_Flist = open(features_path, 'r').read().splitlines()
        feat_list = []
        for f in feat_Flist:
            f_struct = picsom_bin_data(os.path.join(self.dataset_root,f)) 
            feat_list.append(np.array(f_struct.get_float_list(-1))[offset:,:].T.astype(theano.config.floatX))
            print feat_list[-1].shape
		    # this is a 4096 x N numpy array of features
        self.features = np.concatenate(feat_list, axis=0)
        print "Combined all the features. Final size is %d %d"%(self.features.shape[0],self.features.shape[1])
    
    if self.hdf5Flag == 1:
        #Because the HDF5 file is currently stored as one feature per row 
        self.img_feat_size = self.features.shape[1]
    else: 
        self.img_feat_size = self.features.shape[0]
        

    self.aux_pres = 0
    self.aux_inp_size = 0
    if params.get('en_aux_inp',0):
        # Load Auxillary input file, one vec per image
        # NOTE: Assuming same order as feature file
        if params['aux_inp_file'].rsplit('.',1)[1] == 'bin':
            f_struct = picsom_bin_data(os.path.join(self.dataset_root,params['aux_inp_file'])) 
            self.aux_inputs = np.array(f_struct.get_float_list(-1))[offset:,:].T.astype(theano.config.floatX) 
        elif params['aux_inp_file'].rsplit('.',1)[1] == 'txt':   
            feat_Flist = open(os.path.join(self.dataset_root,params['aux_inp_file']), 'r').read().splitlines()
            feat_list = []
            for f in feat_Flist:
                f_struct = picsom_bin_data(os.path.join(self.dataset_root,f)) 
                feat_list.append(np.array(f_struct.get_float_list(-1))[offset:,:].T.astype(theano.config.floatX))
                print feat_list[-1].shape
		        # this is a 4096 x N numpy array of features
            self.aux_inputs = np.concatenate(feat_list, axis=0)
        self.aux_pres = 1
        self.aux_inp_size = self.aux_inputs.shape[0]

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)
      if img['split'] != 'train':
        self.split['allval'].append(img)
    
    # Build tables for length based sampling
    lenHist = defaultdict(int)
    self.lenMap = defaultdict(list)
    if(dataset == 'coco'):
        self.min_len = 7
        self.max_len = 27
    elif(dataset == 'lsmdc2015'):
        self.min_len = 1
        self.max_len = 40
    else:
        raise ValueError('ERROR: Dont know how to do len splitting for this dataset')
    
    # Build the length based histogram
    for iid, img in enumerate(self.split['train']): 
      for sid, sent in enumerate(img['sentences']):
        ix = max(min(len(sent['tokens']),self.max_len),self.min_len)
        lenHist[ix] += 1
        self.lenMap[ix].append((iid,sid))

    self.lenCdist = np.cumsum(lenHist.values())

    
  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['video_fid'] if self.use_video_feat == 1 else img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      aux_feature_index =  img['video_fid'] if self.use_video_feat == 2 else img['imgid']
      if self.hdf5Flag == 1:  #If you're reading from an HDF5 File
          img['feat'] = self.features[feature_index, :]
      else: 
          img['feat'] = self.features[:,feature_index]
      if self.aux_pres:
        img['aux_inp'] = self.aux_inputs[:,aux_feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    if self.use_extdata == 0 or random.random()>self.edata_prob:
        sent = random.choice(img['sentences'])
    else:
        sent = random.choice(self.ext_data['images'][random.choice(img['extNNIdx'])]['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out
  
  def sampleImageSentencePairByLen(self, l, split='train'):
    """ sample image sentence pair from a split """
    pair = random.choice(self.lenMap[l])

    img = self.split[split][pair[0]]
    sent = img['sentences'][pair[1]]

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out
  
  def getRandBatchByLen(self,batch_size):
    """ sample image sentence pair from a split """
    
    rn = np.random.randint(0,self.lenCdist[-1])
    for l in xrange(len(self.lenCdist)):
        if rn < self.lenCdist[l]:
            break

    l += self.min_len
    batch = [self.sampleImageSentencePairByLen(l) for i in xrange(batch_size)]
    return batch,l
  
  # Used for CNN evaluator training
  def sampPosNegSentSamps(self,batch_size, mode = 'batchtrain', thresh = 1):
    """ sample image sentence pair from a split """
    batch = []
    if mode == 'batchtrain':
        img_idx = np.random.choice(np.arange(len(self.split['train'])),size=batch_size, replace=False)
        for i in img_idx:
            batch.append({'sentence':random.choice(self.split['train'][i]['sentences']),
                        'image':self._getImage(self.split['train'][i])})
        posSamp = np.arange(batch_size,dtype=np.int32)
    elif mode == 'multimodal_lstm':
        img_idx = np.random.choice(np.arange(len(self.split['train'])), size=batch_size, replace=False)
        img = self._getImage(self.split['train'][img_idx[0]])
        for i in img_idx:
            batch.append({'sentence':random.choice(self.split['train'][i]['sentences'])})
        batch[0]['image'] = img
        posSamp = np.arange(1,dtype=np.int32)
    else:
        img_idx = np.random.choice(np.arange(len(self.split['train'])))
        img = self._getImage(self.split['train'][img_idx])
        for si in self.split['train'][img_idx]['prefOrder'][:batch_size]:
            batch.append({'sentence':self.split['train'][img_idx]['sentences'][si]})
        
        posSamp = np.arange(len(img['candScoresSorted']), dtype=np.int32)[np.array(img['candScoresSorted'])>thresh]
        if len(posSamp) == 0:
            posSamp = np.arange(1, dtype=np.int32)
        
        # To keep all the batches of same size, pad if necessary
        for i in xrange(batch_size - len(self.split['train'][img_idx]['prefOrder'])):
            batch.append({'sentence':self.split['train'][img_idx]['sentences'][-1]})
        # Finally store image feature
        batch[0]['image'] = img
    return batch, posSamp
  
  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100,shuffle = False):
    batch = []
    imglist = self.split[split]
    ixd = np.array([np.repeat(np.arange(len(imglist)),self.capperimage), 
            np.tile(np.arange(self.capperimage),len(imglist))]).T
    if shuffle:
      random.shuffle(ixd)
    for i,idx in enumerate(ixd):
      if max_images >= 0 and i >= max_images: break
      img = imglist[idx[0]]
      out = {}
      out['image'] = self._getImage(img)
      out['sentence'] = self._getSentence(img['sentences'][idx[1]])
      batch.append(out)
      if len(batch) >= max_batch_size:
        yield batch
        batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])
  
def prepare_data(batch, wordtoix, maxlen=None, sentTagMap=None, ixw = None, pos_samp = [], prep_for = 'lstm_gen', rand_negs = 0, rev_sents = 0):
    """Create the matrices from the datasets.

    If maxlen is set, we will cut all sequence to this maximum
    length. In case of CNN we will also extend all the sequences
    to reach this maxlen.

    Prepare masks if needed and in approp form. 

    Allowed values for prep_for are: ['lstm_gen', 'lstm_eval', 'cnn']
    """
    prep_for_cls = prep_for.split('_')[0] 
    prep_for_subcls = prep_for.split('_')[1] if len(prep_for.split('_')) > 1 else ''

    seqs = []
    if pos_samp == []:
      xI = np.row_stack(x['image']['feat'] for x in batch)
    else:
      xI = np.row_stack(batch[i]['image']['feat'] for i in pos_samp)

    for ix,x in enumerate(batch):
      if prep_for_cls == 'lstm':#and prep_for_subcls == 'gen'
        if rev_sents == 0:
            seqs.append([0] + [wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix] + [0])
        else:
            seqs.append([0] + [wordtoix[w] for w in reversed(x['sentence']['tokens']) if w in wordtoix] + [0])
      else:
        # No start symbol required for evaluators!
        seqs.append([wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix] + [0])

    if rand_negs > 0:
        for i in xrange(rand_negs):
            seqs.append(np.random.choice(seqs[0],np.maximum(maxlen,len(seqs[0]))).tolist())
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None and maxlen > 0:
      new_seqs = []
      new_lengths = []
      for l, s in zip(lengths, seqs):        
          if l > maxlen:
              new_seqs.append(s[:maxlen-1]+[0])
              new_lengths.append(maxlen)
          else:
              new_seqs.append(s)
              new_lengths.append(l)
      lengths = new_lengths
      seqs = new_seqs
    
    if not (maxlen is not None and maxlen > 0) or (prep_for_cls == 'lstm'): # and prep_for_subcls != 'eval'):
      maxlen = np.max(lengths)
    
    n_samples = len(seqs)

    xW = np.zeros((maxlen, n_samples)).astype('int64')
    # Masks are only for lstms
    if prep_for_cls == 'lstm':
      if prep_for_subcls == 'gen':
        x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
      else:
        x_mask = np.array(lengths,dtype=np.int64) -1
    
    for idx, s in enumerate(seqs):
      xW[:lengths[idx], idx] = s
      if prep_for_cls == 'cnn':
        xW[lengths[idx]:, idx] = -1
      # Masks are only for lstms
      elif prep_for_cls == 'lstm' and prep_for_subcls == 'gen':
        x_mask[:lengths[idx], idx] = 1.
        if sentTagMap != None:
          for i,sw in enumerate(s):
            if sentTagMap[batch[idx]['sentence']['sentid']].get(ixw[sw],'') == 'JJ':
              x_mask[i,idx] = 2 

    inp_list = [xW]
    if prep_for_cls == 'lstm':
      inp_list.append(x_mask)
    
    inp_list.append(xI)

    if 'aux_inp' in batch[0]['image']:
      if pos_samp == []:
        xAux = np.row_stack(x['image']['aux_inp'] for x in batch)
      else:
        xAux = np.row_stack(batch[i]['image']['aux_inp'] for i in pos_samp)
      inp_list.append(xAux)

    return inp_list, (np.sum(lengths) - n_samples)

def getDataProvider(params):
  """ we could intercept a special dataset and return different data providers """
  assert params['dataset'] in ['flickr8k', 'flickr30k', 'coco', 'lsmdc2015'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(params)

def loadArbitraryFeatures(params, idxes,auxidxes = []):
  
  feat_all = []
  aux_all = []
  if params.get('multi_model',0) == 0:
    params['nmodels'] = 1
    
  for i in xrange(params['nmodels']):
      #----------------------------- Loop -------------------------------------#
      features_path = params['feat_file'][i] if params.get('multi_model',0) else params['feat_file']
      if features_path.rsplit('.',1)[1] == 'mat':
        features_struct = scipy.io.loadmat(features_path)
        features = features_struct['feats'][:,idxes].astype(theano.config.floatX) # this is a 4096 x N numpy array of features
      elif features_path.rsplit('.',1)[1] == 'hdf5':
        #If the file is one of Vik's HDF5 Files
        features_struct = h5py.File(features_path,'r')
        features = features_struct['feats'][idxes,:].astype(theano.config.floatX) # this is a N x 2032128 array of features
      elif features_path.rsplit('.',1)[1] == 'bin':
        features_struct = picsom_bin_data(features_path) 
        features = np.array(features_struct.get_float_list(idxes)).T.astype(theano.config.floatX) # this is a 4096 x N numpy array of features
        print "Working on Bin file now"
      elif features_path.rsplit('.',1)[1] == 'txt':
          #This is for feature concatenation.
          # NOTE: Assuming same order of features in all the files listed in the txt file
          feat_Flist = open(features_path, 'r').read().splitlines()
          feat_list = []
          for f in feat_Flist:
              f_struct = picsom_bin_data(f) 
              feat_list.append(np.array(f_struct.get_float_list(idxes)).T)
              print feat_list[-1].shape
      	  # this is a 4096 x N numpy array of features
          features = np.concatenate(feat_list, axis=0).astype(theano.config.floatX)
          print "Combined all the features. Final size is %d %d"%(features.shape[0],features.shape[1])
      
      aux_inp = []
      aux_inp_file = params['aux_inp_file'][i] if params.get('multi_model',0) else params.get('aux_inp_file',None)
      if aux_inp_file != None:
          # Load Auxillary input file, one vec per image
          # NOTE: Assuming same order as feature file
          auxidxes = idxes if auxidxes == [] else auxidxes
          if aux_inp_file.rsplit('.',1)[1] == 'bin':
              f_struct = picsom_bin_data(aux_inp_file)
              aux_inp = np.array(f_struct.get_float_list(auxidxes)).T.astype(theano.config.floatX)
          elif aux_inp_file.rsplit('.',1)[1] == 'txt':
              feat_Flist = open(aux_inp_file, 'r').read().splitlines()
              feat_list = []
              for f in feat_Flist:
                  f_struct = picsom_bin_data(f)
                  feat_list.append(np.array(f_struct.get_float_list(idxes)).T.astype(theano.config.floatX))
                  print feat_list[-1].shape
	              # this is a 4096 x N numpy array of features
              aux_inp = np.concatenate(feat_list, axis=0)
      
      feat_all.append(features)
      aux_all.append(aux_inp)

  if params.get('multi_model',0) == 0:
    return features, aux_inp
  else:
    return feat_all, aux_all

