# Exported from notebooks/Probing_Classifier_PIE.ipynb.
# Setup and execution notes: README.md.

# %% [original cell 1]
# !pip install transformers

# %% [original cell 2]
import torch
from transformers import GPT2Model, GPT2Tokenizer
from collections import defaultdict
import numpy as np
import random
import json

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# %% [original cell 3]
filename = "MAGPIE_filtered_split_typebased.json"

raw_data = []
for line in open(filename, 'r'):
    raw_data.append(json.loads(line))

print(len(raw_data))

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# %% [original cell 4]
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics
import math

# Data wrangling
def idiomInContext(context, idiom): 
  for sent in context: 
    if idiom in sent: 
      return True
  return False

def apply_contraction_change(s): 
  s = s.replace(" n't", "n't")
  s = s.replace("\n", "")
  s = s.replace(" ‘", "‘")
  s = s.replace(" ’", "’")
  s = s.replace(" '", "'")
  s = s.replace(" , ", ", ")
  s = s.replace(" .", ". ")
  s = s.replace(' ?', '? ')
  s = s.replace(' !', '! ')
  s = s.replace(' - ', '-')
  s = s.replace(' %', '%')
  return s

def contextToStr(context): 
  output = ""
  for c in context: 
    output += apply_contraction_change(c)
  return output.strip()

def getIdiomIndexes(tokenList, idiomTokenList): 
  currIndex = 0
  for token in tokenList: 
    if token.startswith(idiomTokenList[0]): 
      if len(tokenList) >= currIndex + len(idiomTokenList) and tokenList[currIndex + len(idiomTokenList)-1].endswith(idiomTokenList[-1]): 
        return [i for i in range(currIndex, currIndex + len(idiomTokenList))]
      else:
        return []
    currIndex += 1
  return []

def plotTwoHistograms(data1, data2, bins=50): 
  fig, axs = plt.subplots(1, 1, figsize =(7, 4), tight_layout = True)
  axs.hist(data1, bins = bins)
  fig, axs = plt.subplots(1, 1, figsize =(7, 4), tight_layout = True)
  axs.hist(data2, bins = bins)
  plt.show()

def ttest(sample1, sample2): 
  unpooledSE = math.sqrt(statistics.stdev(sample1)**2 / len(sample1) + statistics.stdev(sample2)**2 / len(sample2))
  t_test = stats.ttest_ind(a=sample1, b=sample2, equal_var=False)
  effectSize = round(np.mean(sample1) - np.mean(sample2), 5)
  df = len(sample1) + len(sample2) - 2
  alpha = (1 - 0.95) / 2
  t_star = stats.t.ppf(alpha, df)
  upperBound = round(effectSize - t_star * unpooledSE, 4)
  lowerBound = round(effectSize + t_star * unpooledSE, 4)
  return {"effect size" : effectSize, "confidence interval" : [lowerBound, upperBound], "p-value" : round(t_test.pvalue, 5)}

# %% [original cell 5]
raw_data[1]

# %% [original cell 6]
tokensMin = 10
tokensMax = 400
confidenceThreshold = 0.99

train_idiom = []
train_literal = []
test_idiom = []
test_literal = []
train_idiom_set = set()
train_literal_set = set()
test_idiom_set = set()
test_literal_set = set()

for d in raw_data:
  if d['confidence'] > confidenceThreshold and idiomInContext(d['context'], d['idiom']):
    text = contextToStr(d['context'])
    idiomTokens = tokenizer.tokenize(d['idiom'])
    idiomTokens[0] = "Ġ" + idiomTokens[0]
    tokens = tokenizer.tokenize(text)
    idiomIndexes = getIdiomIndexes(tokens, idiomTokens)
    if len(idiomIndexes) > 0 and len(tokens) < tokensMax and len(tokens) > tokensMin: 
      dic = {}
      dic['context'] = d['context']
      dic['idiom'] = d['idiom']
      dic['idiomIndexes'] = idiomIndexes
      dic['idiomTokens'] = idiomTokens
      dic['tokens'] = tokens
      dic['text'] = text
      # idioms/figurative and training
      if d['label_distribution']['i'] > 0.9 and (d['split'] == 'training' or d['split'] == 'development'):
        train_idiom.append(dic)
        train_idiom_set.add(dic['idiom'])
      # idioms/figurative and test
      elif d['label_distribution']['i'] > 0.9 and d['split'] == 'test':
        test_idiom.append(dic)
        test_idiom_set.add(dic['idiom'])
      # literal and training
      elif d['label_distribution']['l'] > 0.9 and (d['split'] == 'training' or d['split'] == 'development'):
        train_literal.append(dic)
        train_literal_set.add(dic['idiom'])
      # literal and test
      elif d['label_distribution']['l'] > 0.9 and d['split'] == 'test':
        test_literal.append(dic)
        test_literal_set.add(dic['idiom'])
      else:
        continue
        #print(d['label_distribution'], d['split'])
        
trainSize = len(train_idiom)+len(train_literal)
testSize = len(test_idiom)+len(test_literal)
trainPIEs = train_idiom_set.union(train_literal_set)
testPIEs = test_idiom_set.union(test_literal_set)
intsct = trainPIEs.intersection(testPIEs)
print(len(train_idiom), len(train_literal), len(train_idiom)/trainSize)
print(len(test_idiom), len(test_literal), len(test_idiom)/testSize)
print(trainSize/(trainSize+testSize))

print(len(train_idiom_set))
print(len(train_literal_set))
print(len(test_idiom_set))
print(len(test_literal_set))
print(len(trainPIEs))
print(len(testPIEs))
print(intsct)


# %% [original cell 7]
def runAllModels(data, model, tokenizer): 
  # run the model on all data and append the attention matrix and word embedding matrix to the dicts
  count = 0
  for d in data:
    token_ids = tokenizer.convert_tokens_to_ids(d['tokens'])
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
    output = model(tokens_tensor, output_hidden_states=True)
    attn_data_list = output[-1]
    hidden_states = output.hidden_states

    attention_list = [] # 12 layers, 12 heads, n tokens, n tokens
    for layer, attn_data in enumerate(attn_data_list):
      # Process attention
      attn = attn_data[0]
      attention_list.append(attn.tolist())
    d['attn_list'] = attention_list

    embeddings_list = [] # 13 layers, N tokens, embedding of length 768
    # Hidden-states of the model at the output of each layer plus the initial embedding outputs
    for emb in hidden_states:
      embedding = emb[0] 
      embeddings_list.append(embedding.tolist())
    d['emb_matrix'] = embeddings_list
    count += 1
    if count % 500 == 0:
        print(count)

  return data

# %% [original cell 8]
from random import sample

#trainSplit = 0.8
#testSplit = 1 - trainSplit
#idiomSamples = sample(rawdata_idiom, trainSize+testSize)
#literalSamples = sample(rawdata_literal, trainSize+testSize)
#idiomTrainSize = int(math.floor(trainSplit*len(rawdata_idiom)))+1
#idiomTrainSize = int(math.floor(testSplit*len(rawdata_idiom)))+1
#literalTrainSize = int(math.floor(trainSplit*len(rawdata_literal)))
#literalTrainSize = int(math.floor(testSplit*len(rawdata_literal)))
train_idiom_data = runAllModels(train_idiom, model, tokenizer)
print("here")
train_literal_data = runAllModels(train_literal, model, tokenizer)
test_idiom_data = runAllModels(test_idiom, model, tokenizer)
test_literal_data = runAllModels(test_literal, model, tokenizer)

# %% [original cell 9]
from transformers.utils.logging import reset_format
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy
from math import log, e
from numpy import dot
from numpy.linalg import norm
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

np.shape(train_idiom[0]['emb_matrix']) # number of layers, number of tokens, vector length for each token
np.shape(train_idiom[0]['emb_matrix'][3][4])  # word embedding for the 4th token at the 3rd layer

def getCosineSims(idiom_embeddings): 
  allCosineSims = []
  for i in range(len(idiom_embeddings)-1): 
    for j in range(i+1, len(idiom_embeddings)):
      a = idiom_embeddings[i].tolist()[0]
      b = idiom_embeddings[j].tolist()[0]
      cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
      allCosineSims.append(cos_sim)
  res = [np.mean(allCosineSims), max(allCosineSims), min(allCosineSims)]
  return res

# return the average embedding for all idiom tokens with embedding norm appended
def getIdiomEmbedding(embeddings_list, idiomIndexes, layer, allFeatures=True):
  idiomEmbeddings = np.matrix([embeddings_list[layer+1][i] for i in idiomIndexes])
  res = idiomEmbeddings.mean(0).tolist()[0]
  if not allFeatures:  # do not include norm and cosine sim features
    return res
  res.append(np.linalg.norm(res))
  res.extend(getCosineSims(idiomEmbeddings))
  return res  # length 769

def normalizedTokenToToken(attn_list, layer, head, normalized=True): 
  #print(np.shape(attn_list[layer]))
  #tokenToTokenAttn = np.sum(attn_list[layer], axis=0)  # sum across attention heads
  tokenToTokenAttn = np.array(attn_list[layer][head])
  #print(np.shape(tokenToTokenAttn))
  #print(len(tokenToTokenAttn[4]))
  #x1 = tokenToTokenAttn[4]
  #print("axis 0", np.sum(tokenToTokenAttn, axis=0))
  #print("axis 1", np.sum(tokenToTokenAttn, axis=1))
  if normalized == True: 
    tokenToTokenAttn = tokenToTokenAttn / len(tokenToTokenAttn[0])
  #print("axis 0", np.sum(tokenToTokenAttn, axis=0))
  #print("axis 1", np.sum(tokenToTokenAttn, axis=1))
  return tokenToTokenAttn.tolist()

def entropy(arr, base=None): 
  ent = 0.
  # Compute entropy
  base = 2 if base is None else base
  for a in arr:
    if a > 0: 
      ent -= a * log(a, base)
  return ent

def getAttentionEntropyIndexes(attention_list, idiomIndexes):
  # high entropy means attention is spread out, low entropy means attention is targeted
  #print(len(attention_list[0]))
  #print(attention_list[2])
  #x = attention_list[2] / len(attention_list[0])
  ent = [entropy(attention_list[i]) for i in idiomIndexes]
  return np.mean(ent)

def getAttentionFeatures(d, layer, head): 
  """
  return list of avg PIE to context, avg PIE to PIE, and avg context to PIE attentions, and attention entropy
  """
  allAttns = []
  attn_list = d['attn_list']
  tokenToTokenAttn = normalizedTokenToToken(attn_list, layer, head)
  #print(tokenToTokenAttn)
  idiomIndexes = d['idiomIndexes']
  contextIndexes = [ x for x in range(idiomIndexes[-1]+1, len(d['tokens'])) ]
  # context to PIE attention
  contextToPIE = []
  for j in idiomIndexes: 
    for k in contextIndexes: 
      if j <= k: 
        contextToPIE.append(tokenToTokenAttn[k][j])
  if len(contextIndexes) > 0:
    allAttns.append(sum(contextToPIE) / len(contextToPIE))
  else: 
    allAttns.append(0)

  # PIE to PIE attention
  PIEtoPIE = []
  for j in range(len(idiomIndexes)): 
      for k in range(j, len(idiomIndexes)): 
          PIEtoPIE.append(tokenToTokenAttn[idiomIndexes[k]][idiomIndexes[j]])
  allAttns.append(sum(PIEtoPIE) / len(PIEtoPIE))

  # PIE to context attention
  PIEtoContext = []
  for j in idiomIndexes: 
      for k in contextIndexes: 
        if k < j: 
          PIEtoContext.append(tokenToTokenAttn[j][k])
  if len(PIEtoContext) > 0: 
    allAttns.append(sum(PIEtoContext) / len(PIEtoContext))
  else:
    allAttns.append(0)

  # attention entropy from idiom tokens
  allAttns.append(getAttentionEntropyIndexes(tokenToTokenAttn, idiomIndexes))

  return allAttns

def getAllAttentionFeatures(d, layer): 
  """
  loop through all attention heads to get attention features
  """
  allAttns = []
  for i in range(12): 
    allAttns.extend(getAttentionFeatures(d, layer, i))
  return allAttns

def getDesignMatrix(data_idiom, data_literal, layer, embAllFeatures=True, attnFeatures=True):
  """
  return the design matrix for log regression
  each row is a sample
  features included: idiom embedding vector (averaged), average PIE to context, PIE to PIE, context to PIE, and total inward attentions
  """
  X = []
  for d in data_idiom: 
    oneSampleX = getIdiomEmbedding(d['emb_matrix'], d['idiomIndexes'], layer, allFeatures=embAllFeatures)
    if attnFeatures: 
        oneSampleX.extend(getAllAttentionFeatures(d, layer))
    X.append(oneSampleX)

  for d in data_literal: 
    oneSampleX = getIdiomEmbedding(d['emb_matrix'], d['idiomIndexes'], layer, allFeatures=embAllFeatures)
    if attnFeatures:
        oneSampleX.extend(getAllAttentionFeatures(d, layer))
    X.append(oneSampleX)
  #print(np.shape(X))
  return X

def getTestAccuracy(train_idiom, train_literal, test_idiom, test_literal, layer, embAllFeatures, attnFeatures): 
  # format data for log regression
  X = getDesignMatrix(train_idiom, train_literal, layer, embAllFeatures=embAllFeatures, attnFeatures=attnFeatures)
  Y = [1] * len(train_idiom)
  Y.extend([0] * len(train_literal))

  # train model
  #logModel = LogisticRegression(solver='newton-cg', random_state=0, penalty='none')
  logModel = LogisticRegression(solver='liblinear', random_state=0, penalty='l1', C=0.1)
  logModel.fit(X, Y)

  # test model
  X_test = getDesignMatrix(test_idiom, test_literal, layer, embAllFeatures=embAllFeatures, attnFeatures=attnFeatures)
  Y_test = [1] * len(test_idiom)
  Y_test.extend([0] * len(test_literal))
  Y_pred = logModel.predict(X_test)
  Y_pred_prob = logModel.predict_proba(X_test)[::,1]

  return logModel.score(X_test, Y_test), Y_pred, Y_test, Y_pred_prob

# %% [original cell 10]
regParams = [0.001, 0.01, 0.1, 1, 10]
bestModels = {}
for i in range(12): 
    bestAcc = 0
  #for j in regParams:
    # embeddings only
    score, Y_pred, Y_test, Y_pred_prob = getTestAccuracy(train_idiom, train_literal, test_idiom, test_literal, i, embAllFeatures=False, attnFeatures=False)
    print("case 1, layer:", i+1, 'accuracy:', score)
    if score > bestAcc:
        bestModels[i] = [Y_pred, Y_test, Y_pred_prob, round(score, 4), 1]
        bestAcc = score
    # embeddings + norm + cos sim
    score, Y_pred, Y_test, Y_pred_prob = getTestAccuracy(train_idiom, train_literal, test_idiom, test_literal, i, embAllFeatures=True, attnFeatures=False)
    print("case 2, layer:", i+1, 'accuracy:', score)
    if score > bestAcc:
        bestModels[i] = [Y_pred, Y_test, Y_pred_prob, round(score, 4), 2]
        bestAcc = score
    # all (emb and attention features)
    score, Y_pred, Y_test, Y_pred_prob = getTestAccuracy(train_idiom, train_literal, test_idiom, test_literal, i, embAllFeatures=True, attnFeatures=True)
    print("case 3, layer:", i+1, 'accuracy:', score)
    if score > bestAcc:
        bestModels[i] = [Y_pred, Y_test, Y_pred_prob, round(score, 4), 3]
        bestAcc = score

#print(bestModels)

# %% [original cell 11]
def plotROC(Y_test, Y_pred_prob):
    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_prob)
    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
#print(bestModels)
for layer in bestModels.keys():
    Y_test = bestModels[layer][1]
    Y_pred_prob = bestModels[layer][2]
    score = bestModels[layer][3]
    print(layer, score, "case:", bestModels[layer][4])
    plotROC(Y_test, Y_pred_prob)

# %% [original cell 12]

