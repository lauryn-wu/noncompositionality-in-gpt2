# Exported from notebooks/NC_Attention.ipynb.
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
from random import sample

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# %% [original cell 3]
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics
import math
import re

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
  return output

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

# %% [original cell 4]
filename = "en_TaskIndependentData.json"

raw_data = []
f = open(filename, 'r')
data = json.load(f)

for mwe in data['train_zero_shot']:
  for example in mwe:
    if example[8] == 0:
      raw_data.append(example)

for mwe in data['dev']:
  for example in mwe:
    if example[8] == 0:
      raw_data.append(example)

for mwe in data['test']:
  for example in mwe:
    if example[8] == 0:
      raw_data.append(example)

f.close()

random.seed(1)
random.shuffle(raw_data)
filtered_data = []
nc_set = set()
for r_data in raw_data:
    if r_data[1] not in nc_set and r_data[2] != 'None' and r_data[3] != None:
        nc_set.add(r_data[1])
        filtered_data.append(r_data)

print(len(raw_data), len(filtered_data))

# %% [original cell 5]
raw_data[702]

# %% [original cell 6]
class Models:
    def __init__(self, data, model, tokenizer):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.idiom_data = []
        self.literal_data = []
        self.syn_data = []
        self.all_data = []  # models where all there cases (idiom, literal, and synonym) are valid
        self.runAllModels()
        
    def runIdiomModel(self, i): 
        # run the model on all data and append the attention matrix and word embedding matrix to the dicts
        d = self.data[i]
        dic = {}
        text = d[10] + ' ' + d[11] + ' ' + d[12]
        dic['text'] = formatString(text)
        tokens = tokenizer.tokenize(dic['text'])
        dic['tokens'] = tokens
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
            self.idiom_data.append(dic)
            return dic

        return None

    def runLiteralModel(self, i): 
        # run the model on all data and append the attention matrix and word embedding matrix to the dicts
        d = self.data[i]
        dic = {}
        repl = d[2]
        # initializing substring to be replaced
        subs = d[1]
        compiled = re.compile(re.escape(subs), re.IGNORECASE)
        literalStr = compiled.sub(repl, d[11])
        literalStr = d[10] + ' ' + literalStr + ' ' + d[12]
        dic['text'] = formatString(literalStr)
        tokens = tokenizer.tokenize(dic['text'])
        dic['tokens'] = tokens
        dic['idiomIndexes'] = getIdiomIndexes(tokens, repl)
        #print(repl)
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
            self.literal_data.append(dic)
            return dic

        return None

    def runSynonymModel(self, i): 
        # run the model on all data and append the attention matrix and word embedding matrix to the dicts
        d = self.data[i]
        dic = {}
        repl = d[3]
        # initializing substring to be replaced
        subs = d[1]
        compiled = re.compile(re.escape(subs), re.IGNORECASE)
        literalStr = compiled.sub(repl, d[11])
        literalStr = d[10] + ' ' + literalStr + ' ' + d[12]
        dic['text'] = formatString(literalStr)
        tokens = tokenizer.tokenize(dic['text'])
        dic['tokens'] = tokens
        dic['idiomIndexes'] = getIdiomIndexes(tokens, repl)
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
            self.syn_data.append(dic)
            return dic
        
        return None

    def runAllModels(self):
        for i in range(len(self.data)):
            idiomModel = self.runIdiomModel(i)
            literalModel = self.runLiteralModel(i)
            synModel = self.runSynonymModel(i)
            if idiomModel and literalModel and synModel:
                self.all_data.append([idiomModel, literalModel, synModel])
            if i % 30 == 0:
                print(i)

# %% [original cell 7]
models = Models(filtered_data, model, tokenizer)
all_data = models.all_data
idiom_data = models.idiom_data
literal_data = models.literal_data
synonym_data = models.syn_data

# %% [original cell 8]
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from math import log, e
from numpy import dot
from numpy.linalg import norm

class Attention:
    def __init__(self, idiom_data, literal_data, syn_data, layer, head=None):
        self.idiom_data = idiom_data
        self.literal_data = literal_data
        self.synonym_data = syn_data
        self.layer = layer
        self.head = head
        self.allIdiomContextToPIE = []
        self.allLiteralContextToPIE = []
        self.allSynContextToPIE = []
        self.allIdiomPIEToPIE = []
        self.allLiteralPIEToPIE = []
        self.allSynPIEToPIE = []
        self.allIdiomPIEtoContext = []
        self.allLiteralPIEtoContext = []
        self.allSynPIEtoContext = []
        self.allIdiomEnt = []
        self.allLiteralEnt = []
        self.allSynEnt = []
        self.allIdiomEntTo = []
        self.allLiteralEntTo = []
        self.allSynEntTo = []
        self.allIdiomNorms = []
        self.allLiteralNorms = []
        self.allSynNorms = []
        self.allIdiomCS = []
        self.allLiteralCS = []
        self.allSynCS = []
        self.allAttns()

    def normalizedTokenToToken(self, attn_list, normalized=False): 
        #print(np.shape(attn_list[layer]))
        if self.head == None:
            tokenToTokenAttn = np.mean(attn_list[self.layer], axis=0)  # sum across attention heads
        else:
            tokenToTokenAttn = np.array(attn_list[self.layer][self.head])
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

    def getAvgAttentionsByLayer(self, d):
        attentionIdiom = []  # list of average attention to idiom tokens in each layer
        attentionWord = []
        attention_list = normalizedTokenToToken(d['attn_list'], self.layer)
        tokens = d['tokens']
        idiomIndexes = d['idiomIndexes']
        for i in range(12): # loop over each attention head
            for k in range(len(tokens)): 
                attentionToWord = np.mean(np.array(attention_list[layer][i]).transpose()[k])
                if k in idiomIndexes: # attention to idiom token
                    attentionIdiom.append(attentionToWord)
                else: 
                    attentionWord.append(attentionToWord)
        return np.mean(attentionIdiom), np.mean(attentionWord)

    def contextToPIEAttn(self, d, tokenToTokenAttn, idiomIndexes, contextIndexes, adjust=True):
        contextIndexes = [ x for x in range(idiomIndexes[-1]+1, len(d['tokens'])) ]
        # context to PIE attention
        contextToPIE = []
        for j in idiomIndexes: 
            attnList = []
            for k in contextIndexes: 
                if j <= k: 
                    if adjust: 
                        attnWeight = tokenToTokenAttn[k][j] * k
                    else:
                        attnWeight = tokenToTokenAttn[k][j]
                    attnList.append(attnWeight)
            contextToPIE.append(np.mean(attnList))
        return np.mean(contextToPIE)

    def PIEtoPIEAttn(self, d, tokenToTokenAttn, idiomIndexes, adjust=True):
        # PIE to PIE attention
        PIEtoPIE = []
        for j in range(len(idiomIndexes)): 
            attnList = []
            for k in range(j, len(idiomIndexes)):
                if adjust: 
                    attnWeight = tokenToTokenAttn[idiomIndexes[k]][idiomIndexes[j]] * idiomIndexes[k]
                else:
                    attnWeight = tokenToTokenAttn[idiomIndexes[k]][idiomIndexes[j]]
                attnList.append(attnWeight)
            PIEtoPIE.append(np.mean(attnList))
        return np.mean(PIEtoPIE)

    def PIEtoContextAttn(self, d, tokenToTokenAttn, idiomIndexes, contextIndexes, adjust=True):
      # PIE to context attention
        PIEtoContext = []
        for j in idiomIndexes: 
            attnList = []
            for k in contextIndexes: 
                if k < j: 
                    if adjust: 
                        attnWeight = tokenToTokenAttn[j][k] * j
                    else:
                        attnWeight = tokenToTokenAttn[j][k]
                    attnList.append(attnWeight)
            PIEtoContext.append(np.mean(attnList))
        return np.mean(PIEtoContext)
    
    def entropy(self, arr): 
        ent = 0.
        # Compute entropy
        for a in arr:
            if a > 0: 
                ent -= a * log(a)
        return ent

    def getAttentionEntropyIndexes(self, attention_list, idiomIndexes):
        # high entropy means attention is spread out, low entropy means attention is targeted
        ent = [self.entropy(attention_list[i]) for i in idiomIndexes]
        return np.mean(ent)
    
    def getAttentionEntropyToIndexes(self, attention_list, idiomIndexes):
        # high entropy means attention is spread out, low entropy means attention is targeted
        attnToIndex = np.sum(attention_list, axis=0)
        ent = 0.
        # Compute entropy
        for i in idiomIndexes:
            a = attnToIndex[i]
            if a > 0: 
                ent -= a * log(a)
        return ent / len(idiomIndexes)
    
    def embNorm(self, idiomEmbeddings):
        res = idiomEmbeddings.mean(0).tolist()[0]
        return np.linalg.norm(res)
    
    def getCosineSim(self, idiomEmbeddings): 
        if len(idiomEmbeddings) == 1:
            return 0
        allCosineSims = []
        for i in range(len(idiomEmbeddings)-1): 
            for j in range(i+1, len(idiomEmbeddings)):
                a = idiomEmbeddings[i].tolist()[0]
                b = idiomEmbeddings[j].tolist()[0]
                cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
                allCosineSims.append(cos_sim)
        #print(allCosineSims)
        #res = [np.mean(allCosineSims), max(allCosineSims), min(allCosineSims)]
        res = np.mean(allCosineSims)
        return res
    
    def allAttns(self): 
        for d in self.idiom_data: 
            tokenToTokenAttn = self.normalizedTokenToToken(d['attn_list'])
            idiomIndexes = d['idiomIndexes']
            contextAfterIndexes = [ x for x in range(idiomIndexes[-1]+1, len(d['tokens'])) ]
            contextBeforeIndexes = [ x for x in range(0, idiomIndexes[0]) ]
            embeddings_list = d['emb_matrix']
            idiomEmbeddings = np.matrix([embeddings_list[self.layer+1][i] for i in idiomIndexes])
            a1 = self.contextToPIEAttn(d, tokenToTokenAttn, idiomIndexes, contextAfterIndexes)
            a2 = self.PIEtoPIEAttn(d, tokenToTokenAttn, idiomIndexes)
            a3 = self.PIEtoContextAttn(d, tokenToTokenAttn, idiomIndexes, contextBeforeIndexes)
            a4 = self.getAttentionEntropyIndexes(tokenToTokenAttn, idiomIndexes)
            a5 = self.getAttentionEntropyToIndexes(tokenToTokenAttn, idiomIndexes)
            norm = self.embNorm(idiomEmbeddings)
            cosSim = self.getCosineSim(idiomEmbeddings)
            self.allIdiomContextToPIE.append(a1)
            self.allIdiomPIEToPIE.append(a2)
            self.allIdiomPIEtoContext.append(a3)
            self.allIdiomEnt.append(a4)
            self.allIdiomEntTo.append(a5)
            self.allIdiomNorms.append(norm)
            self.allIdiomCS.append(cosSim)
            
        for d in self.literal_data: 
            tokenToTokenAttn = self.normalizedTokenToToken(d['attn_list'])
            idiomIndexes = d['idiomIndexes']
            contextAfterIndexes = [ x for x in range(idiomIndexes[-1]+1, len(d['tokens'])) ]
            contextBeforeIndexes = [ x for x in range(0, idiomIndexes[0]) ]
            embeddings_list = d['emb_matrix']
            idiomEmbeddings = np.matrix([embeddings_list[self.layer+1][i] for i in idiomIndexes])
            a1 = self.contextToPIEAttn(d, tokenToTokenAttn, idiomIndexes, contextAfterIndexes)
            a2 = self.PIEtoPIEAttn(d, tokenToTokenAttn, idiomIndexes)
            a3 = self.PIEtoContextAttn(d, tokenToTokenAttn, idiomIndexes, contextBeforeIndexes)
            a4 = self.getAttentionEntropyIndexes(tokenToTokenAttn, idiomIndexes)
            a5 = self.getAttentionEntropyToIndexes(tokenToTokenAttn, idiomIndexes)
            norm = self.embNorm(idiomEmbeddings)
            cosSim = self.getCosineSim(idiomEmbeddings)
            self.allLiteralContextToPIE.append(a1)
            self.allLiteralPIEToPIE.append(a2)
            self.allLiteralPIEtoContext.append(a3)
            self.allLiteralEnt.append(a4)
            self.allLiteralEntTo.append(a5)
            self.allLiteralNorms.append(norm)
            self.allLiteralCS.append(cosSim)
            
        for d in self.synonym_data: 
            tokenToTokenAttn = self.normalizedTokenToToken(d['attn_list'])
            idiomIndexes = d['idiomIndexes']
            contextAfterIndexes = [ x for x in range(idiomIndexes[-1]+1, len(d['tokens'])) ]
            contextBeforeIndexes = [ x for x in range(0, idiomIndexes[0]) ]
            embeddings_list = d['emb_matrix']
            idiomEmbeddings = np.matrix([embeddings_list[self.layer+1][i] for i in idiomIndexes])
            a1 = self.contextToPIEAttn(d, tokenToTokenAttn, idiomIndexes, contextAfterIndexes)
            a2 = self.PIEtoPIEAttn(d, tokenToTokenAttn, idiomIndexes)
            a3 = self.PIEtoContextAttn(d, tokenToTokenAttn, idiomIndexes, contextBeforeIndexes)
            a4 = self.getAttentionEntropyIndexes(tokenToTokenAttn, idiomIndexes)
            a5 = self.getAttentionEntropyToIndexes(tokenToTokenAttn, idiomIndexes)
            norm = self.embNorm(idiomEmbeddings)
            cosSim = self.getCosineSim(idiomEmbeddings)
            self.allSynContextToPIE.append(a1)
            self.allSynPIEToPIE.append(a2)
            self.allSynPIEtoContext.append(a3)
            self.allSynEnt.append(a4)
            self.allSynEntTo.append(a5)
            self.allSynNorms.append(norm)
            self.allSynCS.append(cosSim)
        
    def allIdiomAndContextAttns(self, data):
        """
        Return list of average attention to idiom and list of average attention to context
        """
        idiomAttention = []
        wordAttention = []
        for layer in range(12):
            print("layer", layer)
            tokenToTokenAttn = self.normalizedTokenToToken(attn_list)
            idiomAttentionOneLayer = []
            wordAttentionOneLayer = []
            for i, d in enumerate(data): 
                sampleIdiomAttention, sampleWordAttention = getAvgAttentionsByLayer(d, layer)
                idiomAttentionOneLayer.append(sampleIdiomAttention)
                wordAttentionOneLayer.append(sampleWordAttention)
        return idiomAttention, wordAttention
    
    def attnMeanDiff(self):
        contextToPIEDiff = np.nanmean(self.allIdiomContextToPIE) - np.nanmean(self.allSynContextToPIE)
        PIEToPIEDiff = np.nanmean(self.allIdiomPIEToPIE) - np.nanmean(self.allSynPIEToPIE)
        PIEToContextDiff = np.nanmean(self.allIdiomPIEtoContext) - np.nanmean(self.allSynPIEtoContext)
        #print(PIEToContextDiff, self.allIdiomPIEtoContext, self.allLiteralPIEtoContext)
        entDiff = np.nanmean(self.allIdiomEnt) - np.nanmean(self.allSynEnt)
        entToDiff = np.nanmean(self.allIdiomEntTo) - np.nanmean(self.allSynEntTo)
        return contextToPIEDiff, PIEToPIEDiff, PIEToContextDiff, entDiff, entToDiff

# %% [original cell 9]
print(np.shape(literal_data))
print(np.shape(filtered_data))

# %% [original cell 10]
# each row is an attention head
# each column is a layer
# compute the difference between figurative and literal attention measure means, figurative-literal
contextToPIE = []
PIEToPIE = []
PIEtoContext = []
ent = []
entTo = []
for layer in range(12):
    print("layer", layer+1)
    layerContextToPIE = []
    layerPIEToPIE = []
    layerPIEtoContext = []
    layerEnt = []
    layerEntTo = []
    for head in range(12):
        attn = Attention(idiom_data, literal_data, synonym_data, layer, head=head)
        contextToPIEDiff, PIEToPIEDiff, PIEToContextDiff, entDiff, entToDiff = attn.attnMeanDiff()
        layerContextToPIE.append(contextToPIEDiff)
        layerPIEToPIE.append(PIEToPIEDiff)
        layerPIEtoContext.append(PIEToContextDiff)
        layerEnt.append(entDiff)
        layerEntTo.append(entToDiff)
    contextToPIE.append(layerContextToPIE)
    PIEToPIE.append(layerPIEToPIE)
    PIEtoContext.append(layerPIEtoContext)
    ent.append(layerEnt)
    entTo.append(layerEntTo)

# %% [original cell 11]
def plotHeatmap(data, title='', y_lab='Attention Head'):
    data = np.array(data).T
    x_ticks = [i for i in range(1, 13)]
    y_ticks = [i for i in range(1, 13)]
    bound = np.max(np.abs(data))
    s = sns.heatmap(data, xticklabels=x_ticks, yticklabels=y_ticks, cmap='coolwarm', vmin=-bound, vmax=bound)
    s.set(title=title, xlabel='Layer', ylabel=y_lab)
    plt.show()
    
plotHeatmap(PIEToPIE, 'Mean Difference in Phrase to Phrase Attention')
plotHeatmap(PIEtoContext, 'Mean Difference in Phrase to Context Attention')
plotHeatmap(contextToPIE, 'Mean Difference in Context to Phrase Attention')
plotHeatmap(ent, 'Mean Difference in Attention Entropy from Phrase')
plotHeatmap(entTo, 'Mean Difference in Attention Entropy to Phrase')

# %% [original cell 12]
idiomContextToPIE = []
literalContextToPIE = []
synContextToPIE = []
idiomPIEToPIE = []
literalPIEToPIE = []
synPIEToPIE = []
idiomPIEtoContext = []
literalPIEtoContext = []
synPIEtoContext = []
idiomEnt = []
literalEnt = []
synEnt = []
idiomEntTo = []
literalEntTo = []
synEntTo = []
idiomNorms = []
literalNorms = []
synNorms = []
idiomCS = []
literalCS = []
synCS = []

for i in range(12):
    print("layer", i+1)
    attn = Attention(idiom_data, literal_data, synonym_data, i)
    idiomContextToPIE.append(attn.allIdiomContextToPIE)
    literalContextToPIE.append(attn.allLiteralContextToPIE)
    synContextToPIE.append(attn.allSynContextToPIE)
    idiomPIEToPIE.append(attn.allIdiomPIEToPIE)
    literalPIEToPIE.append(attn.allLiteralPIEToPIE)
    synPIEToPIE.append(attn.allSynPIEToPIE)
    idiomPIEtoContext.append(attn.allIdiomPIEtoContext)
    literalPIEtoContext.append(attn.allLiteralPIEtoContext)
    synPIEtoContext.append(attn.allSynPIEtoContext)
    idiomEnt.append(attn.allIdiomEnt)
    literalEnt.append(attn.allLiteralEnt)
    synEnt.append(attn.allSynEnt)
    idiomEntTo.append(attn.allIdiomEntTo)
    literalEntTo.append(attn.allLiteralEntTo)
    synEntTo.append(attn.allSynEntTo)
    idiomNorms.append(attn.allIdiomNorms)
    literalNorms.append(attn.allLiteralNorms)
    synNorms.append(attn.allSynNorms)
    #print(len(attn.allSynNorms))
    idiomCS.append(attn.allIdiomCS)
    #print(len(attn.allIdiomCS))
    literalCS.append(attn.allLiteralCS)
    synCS.append(attn.allSynCS)

# %% [original cell 13]
print(len(idiomCS), len(idiomCS[0]))
print(len(literalCS), len(literalCS[0]))
print(len(synCS), len(synCS[0]))
print(len(idiomNorms), len(idiomNorms[0]))
print(len(literalNorms), len(literalNorms[0]))
print(len(synNorms), len(synNorms[0]))

# %% [original cell 14]
def plotBoxplots(data1, data2, data3, ylimlower=0, ylimupper=0.04, title='', type1='Idiomatic', 
                 type2='Literal', type3='Synonym', y_lab='Values'):
    df1 = pd.DataFrame(data1).T
    df2 = pd.DataFrame(data2).T
    df3 = pd.DataFrame(data3).T
    df1['Type'] = type1
    df2['Type'] = type2
    df3['Type'] = type3
    df = pd.concat([df1, df2, df3], axis=0)
    df['index'] = df.index
    data = df.melt(id_vars=['index', 'Type'], var_name='Layer', value_name=y_lab)
    ax = sns.boxplot(x="Layer", y=y_lab, hue="Type", data=data)  # RUN PLOT   
    x_ticks = [i for i in range(1, 13)]
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(ylimlower, ylimupper)
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.clf()
    plt.close()

# %% [original cell 15]
plotBoxplots(idiomContextToPIE, literalContextToPIE, synContextToPIE, ylimlower=0.023, ylimupper=1.38, 
             title='Context to Phrase Attention of Noun Compounds and Replacements', y_lab='Adjusted Attention')
plotBoxplots(idiomPIEToPIE, literalPIEToPIE, synPIEToPIE, ylimlower=0.06, ylimupper=24.555, 
             title='Phrase to Phrase Attention of Noun Compounds and Replacements', y_lab='Adjusted Attention')
plotBoxplots(idiomPIEtoContext, literalPIEtoContext, synPIEtoContext, ylimlower=0.54, ylimupper=1.024, 
             title='Phrase to Context Attention of Noun Compounds and Replacements', y_lab='Adjusted Attention')
plotBoxplots(idiomEnt, literalEnt, synEnt, ylimlower=0.35, ylimupper=4.123, 
             title='Attention Entropy From Noun Compounds and Replacements', y_lab='Adjusted Attention Entropy From')
plotBoxplots(idiomEntTo, literalEntTo, synEntTo, ylimlower=-0.39, ylimupper=0.38, 
             title='Attention Entropy To Noun Compounds and Replacements', y_lab='Adjusted Attention Entropy To')
plotBoxplots(np.log(idiomNorms), np.log(literalNorms), np.log(synNorms), ylimlower=3.7, ylimupper=5.8, 
             title='Log of Embedding Norm of Noun Compounds and Replacements', y_lab='Natural Log of Embedding Norm')

# %% [original cell 16]
plotBoxplots(idiomCS, literalCS, synCS, ylimlower=0.38, ylimupper=1, 
             title='Mean Cosine Sim Within Phrase', y_lab='Cosine Similarity')

# %% [original cell 17]
def cosSimToPrevLayer(data): 
    cosSims = []
    for d in data: 
        cosSimOneSample = []
        idiomIndexes = d['idiomIndexes']
        embeddings_list = d['emb_matrix']
        idiomEmbedding0 = np.matrix([embeddings_list[0][i] for i in idiomIndexes])
        prevEmb = idiomEmbedding0.mean(0).tolist()[0]
        for layer in range(1, 13):
            idiomEmbeddings = np.matrix([embeddings_list[layer][i] for i in idiomIndexes])
            curEmb = idiomEmbeddings.mean(0).tolist()[0]
            cos_sim = np.dot(prevEmb, curEmb)/(np.linalg.norm(prevEmb)*np.linalg.norm(curEmb))
            cosSimOneSample.append(cos_sim)
            prevEmb = curEmb
        cosSims.append(cosSimOneSample)
    return cosSims

idiomCS = cosSimToPrevLayer(idiom_data)
literalCS = cosSimToPrevLayer(literal_data)
synCS = cosSimToPrevLayer(synonym_data)

# %% [original cell 18]
def plotBoxplots2(data1, data2, data3, ylimlower=0, ylimupper=0.005, title='', 
                  type1='Idiomatic', type2='Literal', type3='Synonym', y_lab='Values'):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df1['Type'] = type1
    df2['Type'] = type2
    df3['Type'] = type3
    df = pd.concat([df1, df2, df3], axis=0)
    df['index'] = df.index
    data = df.melt(id_vars=['index', 'Type'], var_name='Layer', value_name=y_lab)
    ax = sns.boxplot(x="Layer", y=y_lab, hue="Type", data=data)
    x_ticks = [i for i in range(1, 13)]
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(ylimlower, ylimupper)
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.clf()
    plt.close()
    
plotBoxplots2(idiomCS, literalCS, synCS, ylimlower=0.05, ylimupper=1, 
              title='Mean Cosine Sim of Phrase to Previous Layer', y_lab='Cosine Similarity')

# %% [raw]
# Compare the Context (Sentence) Cosine Sims Between Idiom to Literal Interpretation and Idiom to Synonym

# %% [original cell 20]
from itertools import chain

def getCosineSimHelperContext(emb1, emb2, indexes1, indexes2, layer): 
    concatEmb1 = list(chain.from_iterable([emb1[layer][i] for i in indexes1]))
    concatEmb2 = list(chain.from_iterable([emb2[layer][i] for i in indexes2]))
    cos_sim = np.dot(concatEmb1, concatEmb2)/(np.linalg.norm(concatEmb1)*np.linalg.norm(concatEmb2))
    return cos_sim

def getCosineSimHelperPhrase(emb1, emb2, indexes1, indexes2, layer):
    phraseEmb1 = np.matrix([emb1[layer][i] for i in indexes1])
    avgEmb1 = phraseEmb1.mean(0).tolist()[0]
    phraseEmb2 = np.matrix([emb2[layer][i] for i in indexes2])
    avgEmb2 = phraseEmb1.mean(0).tolist()[0]
    cos_sim = np.dot(avgEmb1, avgEmb2)/(np.linalg.norm(avgEmb1)*np.linalg.norm(avgEmb2))
    return cos_sim

def contextCosineSim(all_data):
    literalContextCS = []
    synContextCS = []
    count = 0
    for i in range(len(all_data)):
        idiom_d = all_data[i][0]
        literal_d = all_data[i][1]
        synonym_d = all_data[i][2]
        idiomContextInd = [ x for x in range(0, len(idiom_d['tokens'])) if x not in idiom_d['idiomIndexes'] ]
        literalContextInd = [ x for x in range(0, len(literal_d['tokens'])) if x not in literal_d['idiomIndexes'] ]
        synonymContextInd = [ x for x in range(0, len(synonym_d['tokens'])) if x not in synonym_d['idiomIndexes'] ]
        idiomEmb = idiom_d['emb_matrix']
        literalEmb = literal_d['emb_matrix']
        synEmb = synonym_d['emb_matrix']
        literalOneSample = []
        synOneSample = []
        if len(idiomContextInd) == len(literalContextInd) and len(idiomContextInd) == len(synonymContextInd):
            for j in range(12):
                literalCS = getCosineSimHelperContext(idiomEmb, literalEmb, idiomContextInd, literalContextInd, j)
                synCS = getCosineSimHelperContext(idiomEmb, synEmb, idiomContextInd, synonymContextInd, j)
                literalOneSample.append(literalCS)
                synOneSample.append(synCS)
            literalContextCS.append(literalOneSample)
            synContextCS.append(synOneSample)
        else:
            count += 1
        if i % 500 == 0:
            print(i)
    print('num samples with token lengths that do not match:', count)
    return literalContextCS, synContextCS


def NCCosineSim(all_data):
    literalNCCS = []
    synNCCS = []
    for i in range(len(all_data)):
        idiom_d = all_data[i][0]
        literal_d = all_data[i][1]
        synonym_d = all_data[i][2]
        idiomInd = idiom_d['idiomIndexes']
        literalInd = literal_d['idiomIndexes']
        synonymInd = synonym_d['idiomIndexes']
        idiomEmb = idiom_d['emb_matrix']
        literalEmb = literal_d['emb_matrix']
        synEmb = synonym_d['emb_matrix']
        literalOneSample = []
        synOneSample = []
        if len(idiomInd) == len(literalInd) and len(idiomInd) == len(synonymInd):
            for j in range(12):
                literalCS = getCosineSimHelperContext(idiomEmb, literalEmb, idiomInd, literalInd, j)
                synCS = getCosineSimHelperContext(idiomEmb, synEmb, idiomInd, synonymInd, j)
                literalOneSample.append(literalCS)
                synOneSample.append(synCS)
            literalNCCS.append(literalOneSample)
            synNCCS.append(synOneSample)
        if i % 30 == 0:
            print(i)
    return literalNCCS, synNCCS

        
#literalContextCS, synContextCS = contextCosineSim(all_data)
literalNCCS, synNCCS = NCCosineSim(all_data)

# %% [original cell 21]
def plotBoxplots3(data1, data2, ylimlower=0, ylimupper=0.005, type1='Literal', type2='Synonym',
                  title='', y_lab='Cosine Similarity'):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df1['Type'] = 'Literal'
    df2['Type'] = 'Synonym'
    df = pd.concat([df1, df2], axis=0)
    df['index'] = df.index
    data = df.melt(id_vars=['index', 'Type'], var_name='Layer', value_name=y_lab)
    ax = sns.boxplot(x="Layer", y=y_lab, hue="Type", data=data)  # RUN PLOT   
    x_ticks = [i for i in range(1, 13)]
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(ylimlower, ylimupper)
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.clf()
    plt.close()

#plotBoxplots3(literalContextCS, synContextCS, ylimlower=0.9978, ylimupper=1, 
#              title='Cosine Sim of Context Embeddings to Idiomatic NC', type1='Literal', 
#              type2='Synonym', y_lab='Cosine Similarity')
plotBoxplots3(literalNCCS, synNCCS, ylimlower=0.43, ylimupper=1.01, 
              title='Cosine Sim of Replacement Phrase Embeddings to Idiomatic NC', type1='Literal', 
              type2='Synonym', y_lab='Cosine Similarity')

# %% [original cell 22]
overlapLiteral = 0
overlapSyn = 0
for d in raw_data:
    idiom = d[1].lower().split()
    literal = d[2].lower().split()
    synonym = d[3].lower().split()
    for w in literal: 
        if w in idiom:
            overlapLiteral += 1
    for w in synonym:
        if w in idiom:
            overlapSyn += 1

print(overlapLiteral, overlapSyn)

# %% [original cell 23]
