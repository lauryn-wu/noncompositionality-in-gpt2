# Exported from notebooks/Probing_PIE-Masked_Attn.ipynb.
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
from transformers.utils.logging import reset_format
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy
from math import log, e
from numpy import dot
from numpy.linalg import norm
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from random import sample
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics
import math

# mask an attention head
# each list is a layer, each list has 12 heads
HEAD_MASK = torch.ones([12, 12], dtype=torch.float64)
mask_attn_heads = [[4, 7], [5, 7], [8, 7], [10, 7], [6, 0], [9, 0], [10, 0], 
                   [9, 3], [3, 6], [5, 6], [10, 11], [5, 1], [6, 9], [5, 10], 
                  [4, 3], [3, 8], [3, 4], [3, 0], [5, 0], [0, 1], [0, 5], 
                   [1, 11], [3, 7], [4, 4], [2, 5], [3, 10], [5, 4], [8, 5], 
                   [2, 1], [5, 9], [5, 11], [4, 10], [5, 5], [10, 9], 
                   [11, 10], [7, 11], [6, 6], [7, 7], [3, 2], [1, 6], [0, 4], 
                   [6, 8], [8, 9], [2, 2], [3, 3], [7, 8], [3, 5], [4, 5], 
                   [9, 2], [9, 5], [2, 9], [8, 6], [1, 0], [5, 8], [0, 3], 
                   [4, 0], [7, 2], [4, 11], [7, 1], [11, 5]]
print(len(mask_attn_heads))
for tup in mask_attn_heads:
    HEAD_MASK[tup[0]][tup[1]] = 0


class myGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()
        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=HEAD_MASK,
                                  inputs_embeds=inputs_embeds,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        return outputs
        
# initialize tokenizer and model from pretrained GPT2 model with masked attention head
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = myGPT2Model.from_pretrained('gpt2', output_attentions=True)

# %% [original cell 3]
filename = "MAGPIE_filtered_split_typebased.json"

raw_data = []
for line in open(filename, 'r'):
    raw_data.append(json.loads(line))

print(len(raw_data))

# %% [original cell 4]
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

def getIdiomIndexes(tokenList, idiom): 
    idiom_arr = idiom.lower().split()
    idiomStr = ''.join(idiom_arr)
    tokenList = [t.lower() for t in tokenList]
    tokenList = [t[1:] if t[0] == 'ġ' else t for t in tokenList]
    tokenList = [t.replace("'","").replace('"', '') for t in tokenList]
    tokenList = [t for t in tokenList if t != '']
    for i, token in enumerate(tokenList): 
        if idiom_arr[0].startswith(token): # if the current token matches the first token of the idiom
            for j in range(i+1, len(tokenList)+1):
                tokenStr = ''.join(tokenList[i:j])
                if tokenStr.startswith(idiomStr): 
                    return [k for k in range(i, j)]
                if len(tokenStr) > len(idiomStr)+2:
                    break
    #print(tokenList, idiomStr)
    return None

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
tokensMin = 5
tokensMax = 60
confidenceThreshold = 1
tokLenPIE = 4
tokLensBefore = []
tokLensAfter = []
idiomTokLens = []
train_idiom = []
train_literal = []
dev_idiom = []
dev_literal = []
test_idiom = []
test_literal = []
train_idiom_set = set()
train_literal_set = set()
dev_idiom_set = set()
dev_literal_set = set()
test_idiom_set = set()
test_literal_set = set()
indexes = []
j = 0

for d in raw_data:
    if d['confidence'] == confidenceThreshold and idiomInContext(d['context'], d['idiom']):
        text = contextToStr(d['context'])
        idiomStr = d['idiom']
        tokens = tokenizer.tokenize(text)
        idiomIndexes = getIdiomIndexes(tokens, idiomStr)
        if idiomIndexes:
            tokLensBefore.append(len(tokens))
            idiomTokLens.append(len(idiomIndexes))
        if idiomIndexes and len(tokens) > tokensMax:
            startInd = max(0, idiomIndexes[0] - int(tokensMax/2))
            endInd = min(len(tokens), idiomIndexes[-1]+int(tokensMax/2))
            tokens = tokens[startInd:endInd]
            if startInd > 0: 
                idiomIndexes = [i - startInd for i in idiomIndexes]
        if idiomIndexes == None:
            indexes.append(j)
        else:
            tokLensAfter.append(len(tokens))
    
        if idiomIndexes: 
            dic = {}
            dic['idiom'] = d['idiom']
            dic['idiomIndexes'] = idiomIndexes
            dic['tokens'] = tokens
            # idioms/figurative and training
            if d['label_distribution']['i'] > 0.9 and d['split'] == 'training':
                train_idiom.append(dic)
                train_idiom_set.add(d['idiom'])
            # idioms and dev
            elif d['label_distribution']['i'] > 0.9 and d['split']=='development':
                dev_idiom.append(dic)
                dev_idiom_set.add(d['idiom'])
            # idioms/figurative and test
            elif d['label_distribution']['i'] > 0.9 and (d['split'] == 'test' or d['split']=='development'):
                test_idiom.append(dic)
                test_idiom_set.add(d['idiom'])
            
            # literal and training
            elif d['label_distribution']['l'] > 0.9 and d['split'] == 'training':
                train_literal.append(dic)
                train_literal_set.add(d['idiom'])
            elif d['label_distribution']['l'] > 0.9 and d['split']=='development':
                dev_literal.append(dic)
                dev_literal_set.add(d['idiom'])
            # literal and test
            elif d['label_distribution']['l'] > 0.9 and (d['split'] == 'test' or d['split']=='development'):
                test_literal.append(dic)
                test_literal_set.add(d['idiom'])
            else:
                print(d['label_distribution'], d['split'])
                continue
    j += 1
        
trainSize = len(train_idiom)+len(train_literal)
testSize = len(test_idiom)+len(test_literal)
totalSize = trainSize + testSize
trainPIEs = train_idiom_set.union(train_literal_set)
devPIEs = dev_idiom_set.union(dev_literal_set)
testPIEs = test_idiom_set.union(test_literal_set)
print("train idiom set length:", len(train_idiom), "train literal set length:", len(train_literal), len(train_idiom)/trainSize)
print("test idiom set length:", len(test_idiom), "test literal set length:", len(test_literal), len(test_idiom)/testSize)
print("Train Split:", trainSize/totalSize)
print("train_idiom_set types:", len(train_idiom_set))
print("train_literal_set types:", len(train_literal_set))
print("test_idiom_set types:", len(test_idiom_set))
print("test_literal_set types:", len(test_literal_set))
print("idiom types in train set:", len(trainPIEs))
print("idiom types in train set:", len(devPIEs))
print("idiom types in test set:", len(testPIEs))

# %% [original cell 6]
train_idiom_type_intersection = train_idiom_set.intersection(train_literal_set)
dev_idiom_type_intersection = dev_idiom_set.intersection(dev_literal_set)
test_idiom_type_intersection = test_idiom_set.intersection(test_literal_set)
print('idiom types in common between literal and figurative groups in the training set:', 
      len(train_idiom_type_intersection))
print('idiom types in common between literal and figurative groups in the dev set:', 
      len(dev_idiom_type_intersection))
print('idiom types in common between literal and figurative groups in the test set:', 
      len(test_idiom_type_intersection))

def selectIntersectionIdiomTypes(data, intersection):
    # select only idiom types that appear in both literal and figurative data groups
    res = []
    for d in data:
        if d['idiom'] in intersection:
            res.append(d)
    return res

train_literal_filtered = selectIntersectionIdiomTypes(train_literal, train_idiom_type_intersection)
train_idiom_filtered =  selectIntersectionIdiomTypes(train_idiom, train_idiom_type_intersection)
dev_literal_filtered = selectIntersectionIdiomTypes(dev_literal, dev_idiom_type_intersection)
dev_idiom_filtered =  selectIntersectionIdiomTypes(dev_idiom, dev_idiom_type_intersection)
test_literal_filtered =  selectIntersectionIdiomTypes(test_literal, test_idiom_type_intersection)
test_idiom_filtered =  selectIntersectionIdiomTypes(test_idiom, test_idiom_type_intersection)
print("train idiom sample size:", len(train_idiom_filtered), "| train literal sample size:", len(train_literal_filtered))
print("dev idiom sample size:", len(dev_idiom_filtered), "| dev literal sample size:", len(dev_literal_filtered))
print("test idiom sample size:", len(test_idiom_filtered), "| test literal sample size:", len(test_literal_filtered))

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
# sample the idiom group to create a 50/50 split 
random.seed(1)
train_idiom_set = set()
test_idiom_set = set()

train_idiom = sample(train_idiom_filtered, len(train_literal_filtered))
dev_idiom = sample(dev_idiom_filtered, len(dev_literal_filtered))
test_idiom = sample(test_idiom_filtered, len(test_literal_filtered))

train_literal = train_literal_filtered
dev_literal = dev_literal_filtered
test_literal = test_literal_filtered

for d in train_idiom:
    train_idiom_set.add(d['idiom'])
for d in dev_idiom:
    dev_idiom_set.add(d['idiom'])
for d in test_idiom:
    test_idiom_set.add(d['idiom'])
print(len(train_idiom_set.intersection(train_literal_set)))
print(len(dev_idiom_set.intersection(dev_literal_set)))
print(len(test_idiom_set.intersection(test_literal_set)))

train_idiom_data = runAllModels(train_idiom, model, tokenizer)
train_literal_data = runAllModels(train_literal, model, tokenizer)
print("completed training set")
dev_idiom_data = runAllModels(dev_idiom, model, tokenizer)
dev_literal_data = runAllModels(dev_literal, model, tokenizer)
print("completed dev set")
test_idiom_data = runAllModels(test_idiom, model, tokenizer)
test_literal_data = runAllModels(test_literal, model, tokenizer)

# %% [original cell 9]
class Probe:
    def __init__(self, train_idiom, train_literal, dev_idiom, dev_literal, 
                 test_idiom, test_literal, layer, tokLenPIE=None, regCs=[], cases=[1]):
        self.train_idiom = train_idiom
        self.train_literal = train_literal
        self.dev_idiom = dev_idiom
        self.dev_literal = dev_literal
        self.test_idiom = test_idiom
        self.test_literal = test_literal
        self.layer = layer
        self.tokLenPIE = tokLenPIE
        self.regC = regCs
        self.cases = cases
        self.scores = []
        self.Y_pred = []
        self.Y_test = []
        self.Y_pred_prob = 0
        self.maxScore = 0
        self.bestCase = cases[0]
        self.models = []
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.coef = []
        self.fillDfs()
        self.runAllCases()
        
    def getCosineSims(self, idiom_embeddings): 
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
    def getIdiomEmbedding(self, embeddings_list, idiomIndexes):
        idiomEmbeddings = np.matrix([embeddings_list[self.layer+1][i] for i in idiomIndexes])
        res = idiomEmbeddings.mean(0).tolist()[0]
        res.append(np.linalg.norm(res))
        res.extend(self.getCosineSims(idiomEmbeddings))
        return res

    def normalizedTokenToToken(self, attn_list, head, normalized=False): 
        #print(np.shape(attn_list[layer]))
        #tokenToTokenAttn = np.sum(attn_list[layer], axis=0)  # sum across attention heads
        tokenToTokenAttn = np.array(attn_list[self.layer][head])
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

    def entropy(self, arr, base=None): 
      ent = 0.
      # Compute entropy
      base = 2 if base is None else base
      for a in arr:
        if a > 0: 
          ent -= a * log(a, base)
      return ent

    def getAttentionEntropyIndexes(self, attention_list, idiomIndexes):
      # high entropy means attention is spread out, low entropy means attention is targeted
      #print(len(attention_list[0]))
      #print(attention_list[2])
      #x = attention_list[2] / len(attention_list[0])
      ent = [self.entropy(attention_list[i]) for i in idiomIndexes]
      return np.mean(ent)

    def PIEtoPIEAttn(self, d):
        PIEtoPIE = []
        idiomIndexes = d['idiomIndexes']
        attn_list = d['attn_list']
        for head in range(12):
            tokenToTokenAttn = self.normalizedTokenToToken(attn_list, head)
            #print(np.shape(tokenToTokenAttn), len(idiomIndexes))
            for j in range(len(idiomIndexes)): 
                for k in range(j, len(idiomIndexes)): 
                    PIEtoPIE.append(tokenToTokenAttn[idiomIndexes[k]][idiomIndexes[j]])
        return PIEtoPIE

    def getAttentionFeatures(self, d, head): 
        """
        return list of avg PIE to context, avg PIE to PIE, and avg context to PIE attentions, and attention entropy
        """
        allAttns = []
        attn_list = d['attn_list']
        tokenToTokenAttn = self.normalizedTokenToToken(attn_list, self.layer, head)
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
        allAttns.append(self.getAttentionEntropyIndexes(tokenToTokenAttn, idiomIndexes))

        return allAttns

    def getAllAttentionFeatures(self, d): 
        """
        loop through all attention heads to get attention features
        """
        allAttns = []
        for i in range(12): 
            allAttns.extend(self.getAttentionFeatures(d, i))
        allAttns = [0 if x != x else x for x in allAttns]
        return allAttns
    
    def fillDfs(self):
        self.df_train = self.fillDfHelper(self.train_idiom, self.train_literal)
        self.df_dev = self.fillDfHelper(self.dev_idiom, self.dev_literal)
        self.df_test = self.fillDfHelper(self.test_idiom, self.test_literal)
        
    def fillDfHelper(self, idiomData, literalData):
        X = []
        for d in idiomData: 
            if self.tokLenPIE == None or len(d['idiomIndexes']) == self.tokLenPIE:
                oneSampleX = self.getIdiomEmbedding(d['emb_matrix'], d['idiomIndexes'])  # length of 772
                oneSampleX.extend(self.getAllAttentionFeatures(d))  # length of 844 for 3 tokens
                X.append(oneSampleX)
        for d in literalData: 
            if self.tokLenPIE == None or len(d['idiomIndexes']) == self.tokLenPIE:
                oneSampleX = self.getIdiomEmbedding(d['emb_matrix'], d['idiomIndexes'])
                oneSampleX.extend(self.getAllAttentionFeatures(d))
                X.append(oneSampleX)
        #print('Is Nan', np.any(np.isnan(X)))
        Y = [1] * len(idiomData)
        Y.extend([0] * len(literalData))
        df = pd.DataFrame(X)
        df['Y'] = Y
        return df
    
    def getXsByCase(self, case):
        # only idiom embeddings: col 0-767
        # idiom embeddings and norm and cos sims: cols 0-771
        # attention: cols 772-x (x = 771+6*tokLenPIE*(tokLenPIE+1))
        if self.tokLenPIE != None:
            x = 772+6*self.tokLenPIE*(self.tokLenPIE+1)
        else:
            x = 772+4*12
        # embeddings only
        if case == 1:
            features = [i for i in range(768)]
        # embeddings + norm + cos sim 
        elif case == 2:
            features = [i for i in range(772)]
        # all (emb and attention features)
        elif case == 3:
            features = [i for i in range(x)]
        # only attention features
        elif case == 4:
            features = [i for i in range(771, x)]
        # contextual embedding and attention only
        else:
            f1 = [i for i in range(768)]
            f2 = [i for i in range(771, x)]
            features = f1+f2
        X = self.df_train[features].values
        X_dev = self.df_dev[features].values
        X_test = self.df_test[features].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X_dev = sc.fit_transform(X_dev)
        X_test = sc.transform(X_test)
        return X, X_dev, X_test
            
    def getTestAccuracy2(self, X, X_dev, X_test, Y, Y_dev, Y_test): 
        """
        For a particular case, try all regularization lambdas 
        Return best logistic regression model
        """
        # train model
        maxcv_score, bestY_pred, bestY_test, bestY_pred_prob, coef = -1, 0, 0, 0, 0
        bestC = self.regC[0]
        for c in self.regC:
            logModel = LogisticRegression(solver='saga', tol=1e-3, max_iter=200, random_state=0, penalty='l2', C=c)
            logModel.fit(X, Y)
            
            Y_pred = logModel.predict(X_dev)
            Y_pred_prob = logModel.predict_proba(X_dev)[::,1]
            cv_score = logModel.score(X_dev, Y_dev)
            if cv_score > maxcv_score:
                bestModel = logModel
                maxcv_score = cv_score
                bestC = c

        Y_pred = bestModel.predict(X_test)
        score = bestModel.score(X_test, Y_test)
        Y_pred_prob = bestModel.predict_proba(X_test)[::,1]
        coef = bestModel.coef_[0]
        
        self.scores.append(score)
        self.models.append(bestModel)
        return score, Y_pred, Y_test, bestY_pred_prob, bestC, coef
    
    def getTestAccuracy(self, X, X_test, Y, Y_test): 
        """
        For a particular case, try all regularization lambdas 
        Return best logistic regression model
        """
        # train model
        maxScore, bestY_pred, bestY_test, bestY_pred_prob, coef = 0, 0, 0, 0, 0
        logModel = LogisticRegressionCV(cv=5, tol=0.001, solver='saga', max_iter=200, random_state=0, penalty='l2')
        logModel.fit(X, Y)
        coef = logModel.coef_[0]
        bestC = logModel.C_[0]
        # test model
        Y_pred = logModel.predict(X_test)
        Y_pred_prob = logModel.predict_proba(X_test)[::,1]
        trainScore = logModel.score(X, Y)
        score = logModel.score(X_test, Y_test)
        
        self.scores.append(score)
        self.models.append(logModel)
        return score, Y_pred, Y_test, Y_pred_prob, bestC, coef
    
    def plotCM(self, y_test, y_pred):
        cm = metrics.confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
        
    def plotVarImportance(self, coef):
        # plot feature importance
        mat = np.reshape(coef, (12, int(len(coef)/12)))
        agg_coef = np.mean(mat, axis=0)  # by token to token attention
        agg_coef2 = np.mean(mat, axis=1) # by attention head
        fig, axs =  plt.subplots(1,3, figsize =(16, 4), tight_layout = True)
        axs[0].bar([x for x in range(len(coef))], coef)
        axs[1].bar([x for x in range(len(agg_coef))], agg_coef)
        axs[2].bar([x for x in range(len(agg_coef2))], agg_coef2)
        plt.show()
        
    def runAllCases(self):
        print("running cases")
        for case in self.cases:
            X, X_dev, X_test = self.getXsByCase(case)
            Y = self.df_train['Y'].values
            Y_dev = self.df_dev['Y'].values
            Y_test = self.df_test['Y'].values
            score, bestY_pred, bestY_test, bestY_pred_prob, bestC, coef = self.getTestAccuracy2(X, 
                                                                   X_dev, X_test, Y, Y_dev, Y_test)
            self.coef = coef
            if score > self.maxScore: 
                self.Y_pred = bestY_pred
                self.Y_test = bestY_test
                self.Y_pred_prob = bestY_pred_prob
                self.maxScore = round(score, 4)
                self.bestCase = case
            print("case:", case, 'accuracy:', round(score, 5), "best C:", round(bestC, 3))

# %% [original cell 10]
print(len(train_idiom_data))
print(len(train_literal_data))
print(len(test_idiom_data))
print(len(test_literal_data))

# %% [original cell 11]
regParams = [0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 1, 10]
cases = [1, 4, 5]
allScores = []
allCoefs = []
allYPreds = []
allYTests = []
for i in range(12): 
    print("layer", i+1)
    probe = Probe(train_idiom_data, train_literal_data, dev_idiom_data, dev_literal_data, 
                  test_idiom_data, test_literal_data, layer=i, regCs=regParams, cases=cases)
    allScores.append(probe.scores)
    allCoefs.append(probe.coef)
    allYPreds.append(probe.Y_pred)
    allYTests.append(probe.Y_test)

# %% [original cell 12]
case1 = [score[0] for score in allScores]
case4 = [score[1] for score in allScores]
case5 = [score[2] for score in allScores]

plt.figure(figsize=(7, 5))
x = [i for i in range(1, 13)]
plt.plot(x, case1, label = "emb", marker='o')
#plt.plot(x, case2, label = "case 2", marker='o')
#plt.plot(x, case3, label = "case 3", marker='o')
plt.plot(x, case4, label = "attn", marker='o')
plt.plot(x, case5, label = "emb+attn", marker='o')
plt.xlim(0.5, 12.5)
plt.locator_params(axis='x', nbins=12)
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title('PIE Usage Task Accuracy - All Token Lengths')
plt.legend()
plt.show()

# %% [original cell 13]
print('case1=', case1)
#print(case2)
#print(case3)
print('case4=', case4)
print('case5=', case5)

# %% [original cell 14]
"""
def plotCM(y_tests, y_preds, scores):
    for i in range(len(y_tests)):
        cm = metrics.confusion_matrix(y_tests[i], y_preds[i])
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(scores[i])
        plt.title(all_sample_title, size = 10)

plotCM(allYTests, allYPreds, case2)
"""

# %% [original cell 15]
"""
def plotVarImportanceCase12(coefs):
    for i, coef in enumerate(coefs):
        print("Layer", i+1)
        # plot feature importance
        coef = np.exp(coef)
        #mat = np.reshape(coef, (12, int(len(coef)/12)))
        coef2 = coef[-4:]
        fig, axs =  plt.subplots(1,2, figsize =(16, 4), tight_layout = True)
        axs[0].bar([x for x in range(len(coef))], coef)
        axs[1].bar([x for x in range(len(coef2))], coef2)  # embedding norm, mean cos sim, max cos sim, min cos sim
        plt.show()
        
plotVarImportanceCase12(allCoefs)
"""

# %% [original cell 16]
"""
def plotVarImportance(coefs):
    for i, coef in enumerate(coefs):
        print("Layer", i)
        # plot feature importance
        #coef = np.exp(coef)
        mat = np.reshape(coef, (12, int(len(coef)/12)))
        agg_coef = np.mean(mat, axis=0)  # by token to token attention
        agg_coef2 = np.mean(mat, axis=1) # by attention head
        fig, axs =  plt.subplots(1,3, figsize =(16, 4), tight_layout = True)
        axs[0].bar([x for x in range(len(coef))], coef)
        axs[1].bar([x for x in range(len(agg_coef))], agg_coef)
        axs[2].bar([x for x in range(len(agg_coef2))], agg_coef2)
        plt.show()

plotVarImportance(allCoefs)
"""

# %% [original cell 17]
"""
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
"""

# %% [original cell 18]
