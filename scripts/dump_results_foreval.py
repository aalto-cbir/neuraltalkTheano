import argparse
import json
from collections import defaultdict
import os.path
import re

def buildDbid2Idx(lblfile):
  labels = open(lblfile).read().splitlines()
  dbid2idx = [defaultdict(list)] * 4
  cidx = [1]*4
  for lbl in labels:
      idx,dbStr = lbl.split()
      idx = int(idx[1:])
      splt = int(dbStr[1])
      dbid = dbStr[1:-1]
      if ':' not in dbid:
        dbid2idx[splt][dbid] = cidx[splt]
        cidx[splt] += 1
  
  return dbid2idx


def buildTransDict(dictFile):
  dictVal= open(dictFile,'r').read().splitlines()
  trDict = {}
  for lns in dictVal:
      targ = lns.split()[0]
      for wrdSeq in lns.split()[1:]:
        key = ' '.join(wrdSeq.split('_'))
        trDict[key] = targ
      #trDict[key] = {} 
      #for wrdSeq in lns.split()[1:]:
      #    if wrdSeq.split('_')[0] not in trDict[key]:
      #        trDict[key][wrdSeq.split('_')[0]] = {'lens':set([len(wrdSeq.split('_'))]), 'ext':set([wrdSeq])}
      #    else:
      #        trDict[key][wrdSeq.split('_')[0]]['lens'].add(len(wrdSeq.split('_')))
      #        trDict[key][wrdSeq.split('_')[0]]['ext'].add(wrdSeq)
  
  trDict = dict((re.escape(k), v) for k, v in trDict.iteritems())
  pattern = re.compile("|".join(trDict.keys()))
  return pattern,trDict 
  

def main(params):
  
  testRes = json.load(open(params['resFile'],'r'))
  
  outfile = 'captions_%s_%s_results.json'%(params['target_split'],params['algname'])
  outdir = 'eval/mseval/results' if params['target_db'] == 'coco' else 'eval/lsmdcEval/results/'

  # Build dbid to test dump index
  if params['target_db'] == 'lsmdc2015':
    dbid2idx = buildDbid2Idx(params['labelsFile'])
  
  if params['translate'] == 1:
    pattern,trDict = buildTransDict(params['transdict'])
  
  testResDump = []
  
  for i,img in enumerate(testRes['imgblobs']):
    if params['translate'] == 0:
        txt = img['candidate']['text']
    else:
        txt = pattern.sub(lambda m: trDict[re.escape(m.group(0))],img['candidate']['text'])
        testRes['imgblobs'][i]['candidate']['text'] = txt

    if params['target_db'] == 'coco':
        testResDump.append({'caption': txt,'image_id': int(img['img_path'].split('/')[-1].split('_')[-1].split('.')[0])})
    elif params['target_db'] == 'lsmdc2015_picsom':
        testResDump.append({'caption': txt,'video_id': int(img['img_path'].split('/')[-1].split(':')[0])})
    elif params['target_db'] == 'lsmdc2015':
        dbid = img['img_path'].split('/')[-1].split(':')[0]
        splt = 2 if params['target_split'] == 'test2015' else 1
        testResDump.append({'caption': txt,'video_id': dbid2idx[splt][dbid]})
    else:
        raise ValueError('Error: this db is not handled')

  json.dump(testResDump, open(os.path.join(outdir,outfile), 'w'))
  if params['writeback'] != '':
    json.dump(testRes, open(params['writeback'], 'w'))
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('resFile', type=str, help='the input resultDump')
  parser.add_argument('-td', '--target_db', type=str, default='coco', help='target database to evaluate on')
  parser.add_argument('-ts', '--target_split', type=str, default='val2014', help='target database to evaluate on')
  parser.add_argument('-a', '--algname', type=str, default='fakecap', help='algorithm name to use in output filename')
  parser.add_argument('--translate', type=int, default=0, help='Apply language translation via dict')
  parser.add_argument('--transdict', type=str, default='data/lsmdc2015/commons.dict', help='translation dictionary')
  parser.add_argument('--labelsFile',dest='labelsFile', type=str, default='data/lsmdc2015/labels.txt', help='labels file mapping picsom id to sequence id')
  parser.add_argument('--writeback',dest='writeback', type=str, default='', help='writeback the result struct after translation')

  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
