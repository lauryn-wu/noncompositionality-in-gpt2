# Exported from notebooks/Noun_Compounds.ipynb.
# Setup and execution notes: README.md.

# %% [markdown]
# Test Noun Compounds
# Idiom vs. Literal Translation of the Idiom

# %% [original cell 2]
# !pip install transformers

# %% [original cell 3]
import torch
from transformers import GPT2Model, GPT2Tokenizer
from collections import defaultdict
import numpy as np
import random
import json
import pandas as pd

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# %% [markdown]
# Each Data Example
# 0. idiom id
# 1. MWE
# 2. Literal Meaning
# 3. Non-literal meaning 1 (or None)
# 4. Non-literal meaning 2 (or None)
# 5. Non-literal meaning 3 (or None)
# 6. Proper Noun (As all MWEs can be used as proper nouns)
# 7. Meta Usage (As all MWEs can be used in this way - please see paper for details)
# 8. 0/1 (1 if this example is not idiomatic, i.e. 1 includes Proper noun and Meta usage)
# 9. The fine grained label associated with this example.
# 10. The sentence prior to the target sentence containing the MWE.
# 11. The sentence containing the MWE
# 12. The sentence after the target sentence containing the MWE.
# 13. The source of these sentences.

# %% [original cell 5]
filename = "en_TaskIndependentData.json"

raw_data = []
f = open(filename, 'r')
data = json.load(f)
print(data.keys())
trainData = []
testData = []
devData = []

for mwe in data['train_zero_shot']:
  for example in mwe:
    if example[8] == 0:
      trainData.append(example)

for mwe in data['dev']:
  for example in mwe:
    if example[8] == 0:
      devData.append(example)

for mwe in data['test']:
  for example in mwe:
    if example[8] == 0:
      testData.append(example)

f.close()

# %% [original cell 6]
print(len(trainData))
print(len(devData))
print(len(testData))

# %% [original cell 7]
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics
import math
import re

# Data wrangling
def idiomInContext(context, idiom): 
  for sent in context: 
    if idiom in sent: 
      return True
  return False

def formatString(s): 
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
  return s.strip()

def getIdiomIndexes(tokenList, idiomTokenList): 
    for i, token in enumerate(tokenList): 
        if token==idiomTokenList[0] and len(tokenList) >= i + len(idiomTokenList): 
            if idiomTokenList == tokenList[i:i+len(idiomTokenList)]:
                return [j for j in range(i, i+len(idiomTokenList))]
    idiomTokenList[0] = "Ġ" + idiomTokenList[0]
    for i, token in enumerate(tokenList): 
        if token==idiomTokenList[0] and len(tokenList) >= i + len(idiomTokenList): 
            if idiomTokenList == tokenList[i:i+len(idiomTokenList)]:
                return [j for j in range(i, i+len(idiomTokenList))]
    print(tokenList, idiomTokenList)
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
    return None

def runAllModels(data, model, tokenizer): 
  # run the model on all data and append the attention matrix and word embedding matrix to the dicts
  count = 0
  res = []

  for d in data:
    dic = {}
    dic['text'] = formatString(d[11])
    tokens = tokenizer.tokenize(dic['text'])
    dic['idiomIndexes'] = getIdiomIndexes(tokens, d[1])
    if dic['idiomIndexes'] != None: 
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
        output = model(tokens_tensor, output_hidden_states=True)
        attn_data_list = output[-1]
        hidden_states = output.hidden_states

        attention_list = [] # 12 layers, 12 heads, n tokens, n tokens
        for layer, attn_data in enumerate(attn_data_list):
          # Process attention
          attn = attn_data[0]
          attention_list.append(attn.tolist())
        dic['attn_list'] = attention_list

        embeddings_list = [] # 13 layers, N tokens, embedding of length 768
        # Hidden-states of the model at the output of each layer plus the initial embedding outputs
        for emb in hidden_states:
          embedding = emb[0] 
          embeddings_list.append(embedding.tolist())
        dic['emb_matrix'] = embeddings_list

        count += 1
        if count % 500 == 0:
            print(count)
        res.append(dic)

  return res

def runAllModelsLiteral(data, model, tokenizer): 
  # run the model on all data and append the attention matrix and word embedding matrix to the dicts
  count = 0
  res = []

  for d in data:
    dic = {}
    repl = d[2]
    # initializing substring to be replaced
    subs = d[1]
    compiled = re.compile(re.escape(subs), re.IGNORECASE)
    literalStr = compiled.sub(repl, d[11])
    dic['text'] = formatString(literalStr)
    tokens = tokenizer.tokenize(dic['text'])
    dic['idiomIndexes'] = getIdiomIndexes(tokens, d[2])
    if dic['idiomIndexes'] != None: 
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
        output = model(tokens_tensor, output_hidden_states=True)
        attn_data_list = output[-1]
        hidden_states = output.hidden_states

        attention_list = [] # 12 layers, 12 heads, n tokens, n tokens
        for layer, attn_data in enumerate(attn_data_list):
          # Process attention
          attn = attn_data[0]
          attention_list.append(attn.tolist())
        dic['attn_list'] = attention_list

        embeddings_list = [] # 13 layers, N tokens, embedding of length 768
        # Hidden-states of the model at the output of each layer plus the initial embedding outputs
        for emb in hidden_states:
          embedding = emb[0] 
          embeddings_list.append(embedding.tolist())
        dic['emb_matrix'] = embeddings_list

        count += 1
        if count % 500 == 0:
            print(count)

        res.append(dic)

  return res

# %% [original cell 8]
train_idiom = runAllModels(trainData, model, tokenizer)
print("here")
dev_idiom = runAllModels(devData, model, tokenizer)
test_idiom = runAllModels(testData, model, tokenizer)
train_literal = runAllModelsLiteral(trainData, model, tokenizer)
print("here")
dev_literal = runAllModelsLiteral(devData, model, tokenizer)
test_literal = runAllModelsLiteral(testData, model, tokenizer)

# %% [original cell 9]
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

class Probe:
    def __init__(self, train_idiom, train_literal, dev_idiom, dev_literal, test_idiom, test_literal, 
                 layer, tokLenPIE=None, regCs=[], cases=[1]):
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
        self.df_dev = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.coef = []
        self.fillDfs()
        self.runAllCases()
        
    def getCosineSims(self, idiom_embeddings): 
      if len(idiom_embeddings) == 1:
        return 0
      allCosineSims = []
      for i in range(len(idiom_embeddings)-1): 
        for j in range(i+1, len(idiom_embeddings)):
          a = idiom_embeddings[i].tolist()[0]
          b = idiom_embeddings[j].tolist()[0]
          cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
          allCosineSims.append(cos_sim)
      #print(allCosineSims)
      #res = [np.mean(allCosineSims), max(allCosineSims), min(allCosineSims)]
      res = np.mean(allCosineSims)
      return res

    # return the average embedding for all idiom tokens with embedding norm appended
    def getIdiomEmbedding(self, embeddings_list, idiomIndexes):
        idiomEmbeddings = np.matrix([embeddings_list[self.layer+1][i] for i in idiomIndexes])
        #print(np.shape(idiomEmbeddings))
        res = idiomEmbeddings.mean(0).tolist()[0]
        res.append(np.linalg.norm(res))
        res.append(self.getCosineSims(idiomEmbeddings))
        return res

    def normalizedTokenToToken(self, attn_list, head, normalized=False): 
        #print(np.shape(attn_list[layer]))
        #tokenToTokenAttn = np.sum(attn_list[layer], axis=0)  # sum across attention heads
        tokenToTokenAttn = np.array(attn_list[self.layer][head])
        if normalized == True: 
            tokenToTokenAttn = tokenToTokenAttn / len(tokenToTokenAttn[0])
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
                    contextToPIE.append(self.tokenToTokenAttn[k][j])
        if len(contextIndexes) > 0:
            allAttns.append(sum(contextToPIE) / len(contextToPIE))
        else: 
            allAttns.append(0)

      # PIE to PIE attention
        PIEtoPIE = []
        for j in range(len(idiomIndexes)): 
            for k in range(j, len(idiomIndexes)): 
                PIEtoPIE.append(self.tokenToTokenAttn[idiomIndexes[k]][idiomIndexes[j]])
        allAttns.append(sum(PIEtoPIE) / len(PIEtoPIE))

      # PIE to context attention
        PIEtoContext = []
        for j in idiomIndexes: 
            for k in contextIndexes: 
                if k < j: 
                    PIEtoContext.append(self.tokenToTokenAttn[j][k])
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
                #oneSampleX.extend(self.PIEtoPIEAttn(d))  # length of 844 for 3 tokens
                X.append(oneSampleX)
        for d in literalData: 
            if self.tokLenPIE == None or len(d['idiomIndexes']) == self.tokLenPIE:
                oneSampleX = self.getIdiomEmbedding(d['emb_matrix'], d['idiomIndexes'])
                #oneSampleX.extend(self.PIEtoPIEAttn(d))
                X.append(oneSampleX)
                
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
            x = 771+6*self.tokLenPIE*(self.tokLenPIE+1)
        # embeddings only
        if case == 1:
            features = [i for i in range(768)]
        # embeddings + norm + cos sim 
        elif case == 2:
            features = [i for i in range(770)]
        # all (emb and attention features)
        elif case == 3:
            features = [i for i in range(x+1)]
        # only attention features
        elif case == 4:
            features = [i for i in range(770, x+1)]
        # contextual embedding and attention only
        else:
            f1 = [i for i in range(769)]
            f2 = [i for i in range(770, x+1)]
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
        maxcv_score, bestY_pred, bestY_test, bestY_pred_prob, coef = 0, 0, 0, 0, 0
        bestModel = None
        bestC = self.regC[0]
        for c in self.regC:
            #logModel = LogisticRegression(solver='newton-cg', random_state=0, penalty='none')
            logModel = LogisticRegression(solver='saga', tol=1e-3, max_iter=200, random_state=0, penalty='l2', C=c)
            logModel.fit(X, Y)

            # cross validation of model
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
        #logModel = LogisticRegression(solver='newton-cg', random_state=0, penalty='none')
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
            score, bestY_pred, bestY_test, bestY_pred_prob, bestC, coef = self.getTestAccuracy2(X, X_dev, X_test, Y, Y_dev, Y_test)
            #if case == 4:
            self.coef = coef
                #self.plotVarImportance(coef)
            if score > self.maxScore: 
                self.Y_pred = bestY_pred
                self.Y_test = bestY_test
                self.Y_pred_prob = bestY_pred_prob
                self.maxScore = round(score, 4)
                self.bestCase = case
            print("case:", case, 'accuracy:', round(score, 5), "best C:", round(bestC, 3))

# %% [original cell 10]
regParams = [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5]
cases = [1, 2]
allScores = []
allCoefs = []
allYPreds = []
allYTests = []
for i in range(12): 
    print("layer", i+1)
    probe = Probe(train_idiom, train_literal, dev_idiom, dev_literal, test_idiom, test_literal, layer=i, 
                  regCs=regParams, cases=cases)
    allScores.append(probe.scores)
    allCoefs.append(probe.coef)
    allYPreds.append(probe.Y_pred)
    allYTests.append(probe.Y_test)

# %% [original cell 11]
case1 = [score[0] for score in allScores]
case2 = [score[1] for score in allScores]
#case5 = [score[2] for score in allScores]

plt.figure(figsize=(7, 5))
x = [i for i in range(1, 13)]
plt.plot(x, case1, label = "case 1", marker='o')
plt.plot(x, case2, label = "case 2", marker='o')
#plt.plot(x, case3, label = "case 3", marker='o')
#plt.plot(x, case4, label = "case 4", marker='o')
#plt.plot(x, case5, label = "case 5", marker='o')
plt.xlim(0.5, 12.5)
plt.locator_params(axis='x', nbins=12)
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [original cell 12]
print(case1)
print(case2)

# %% [original cell 13]
def plotVarImportanceCase12(coefs):
    for i, coef in enumerate(coefs):
        print("Layer", i+1)
        # plot feature importance
        coef = np.exp(coef)
        #mat = np.reshape(coef, (12, int(len(coef)/12)))
        coef2 = coef[-2:]
        fig, axs =  plt.subplots(1,2, figsize =(16, 4), tight_layout = True)
        axs[0].bar([x for x in range(len(coef))], coef)
        axs[1].bar([x for x in range(len(coef2))], coef2)  # embedding norm, mean cos sim, max cos sim, min cos sim
        plt.show()
        
plotVarImportanceCase12(allCoefs)

# %% [original cell 14]
