# Exported from notebooks/PIE.ipynb.
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
filename = "MAGPIE_filtered_split_typebased.json"

raw_data = []
for line in open(filename, 'r'):
    raw_data.append(json.loads(line))

random.seed(1)
random.shuffle(raw_data)
print(len(raw_data))

# %% [original cell 4]
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

def getIdiomIndexes(tokenList, idiomTokenList): 
  indexes = []
  for token in idiomTokenList: 
    if token in tokenList: 
      indexes.append(tokenList.index(token))
  return indexes

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

# %% [original cell 5]
#tokensMin = 5
tokensMax = 100
confidenceThreshold = 1
tokLenPIE = 4
tokLensBefore = []
tokLensAfter = []
idiomTokLens = []
idiom = []
literal = []
idiom_set = set()
literal_set = set()
indexes = []
j = 0
random.seed(1)
random.shuffle(raw_data)

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
            dic['idiomIndexes'] = idiomIndexes
            dic['tokens'] = tokens
            dic['idiom'] = d['idiom']
          # idioms/figurative and training
            if d['label_distribution']['i'] > 0.9:
                idiom.append(dic)
                idiom_set.add(idiomStr)
            if d['label_distribution']['l'] > 0.9:
                literal.append(dic)
                literal_set.add(idiomStr)
        j += 1

print(len(idiom_set), len(literal_set))
# select one of each idiom type, where the idiom type can be idiomatic and literal 
idiom_type_intersection = idiom_set.intersection(literal_set)

idiom_set = set()
literal_set = set()
idiom_data = []
literal_data = []
for d in idiom:
    if d['idiom'] not in idiom_set and d['idiom'] in idiom_type_intersection:
        idiom_data.append(d)
        idiom_set.add(d['idiom'])
        
for d in literal:
    if d['idiom'] not in literal_set and d['idiom'] in idiom_type_intersection:
        literal_data.append(d)
        literal_set.add(d['idiom'])

print("idiom sample size:", len(idiom_data))
print("literal sample size:", len(literal_data))
print("idiom set size:", len(idiom_set))
print("literal set size:", len(literal_set))

# %% [original cell 6]
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
    if count % 100 == 0:
        print(count)

  return data

idiom_data = runAllModels(idiom_data, model, tokenizer)
print("here")
literal_data = runAllModels(literal_data, model, tokenizer)

# %% [original cell 7]
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from math import log, e
from numpy import dot
from numpy.linalg import norm

class Attention:
    def __init__(self, idiom_data, literal_data, layer, head=None, firstContextIndex=0):
        self.idiom_data = idiom_data
        self.literal_data = literal_data
        self.layer = layer
        self.head = head
        self.firstContextIndex = firstContextIndex #whether or not to filter out the first token
        self.allIdiomContextToPIE = []
        self.allLiteralContextToPIE = []
        self.allIdiomPIEToPIE = []
        self.allLiteralPIEToPIE = []
        self.allIdiomPIEtoContext = []
        self.allLiteralPIEtoContext = []
        self.allIdiomEnt = []
        self.allLiteralEnt = []
        self.allIdiomEntTo = []
        self.allLiteralEntTo = []
        self.allIdiomNorms = []
        self.allLiteralNorms = []
        self.allIdiomCS = []
        self.allLiteralCS = []
        self.allAttns()

    def normalizedTokenToToken(self, attn_list, normalized=False, adjust=False): 
        #print(np.shape(attn_list[layer]))
        if self.head == None:
            tokenToTokenAttn = np.mean(attn_list[self.layer], axis=0)  # sum across attention heads
        else:
            tokenToTokenAttn = np.array(attn_list[self.layer][self.head])
        #tokenToTokenAttn2 = np.sum(attn_list[self.layer], axis=0)
        #print('here1', np.shape(tokenToTokenAttn))
        #print('here2', np.shape(tokenToTokenAttn2))
        #print(len(tokenToTokenAttn[4]))
        #x1 = tokenToTokenAttn[4]
        #print("axis 0", np.sum(tokenToTokenAttn, axis=0))
        #print("axis 1", np.sum(tokenToTokenAttn, axis=1))
        if normalized == True: 
            tokenToTokenAttn = tokenToTokenAttn / len(tokenToTokenAttn[0])
        if adjust == True:
            for i in range(tokenToTokenAttn.shape[0]):
                for j in range(tokenToTokenAttn.shape[1]):
                    tokenToTokenAttn[i][j] = tokenToTokenAttn[i][j] * i
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
            contextBeforeIndexes = [ x for x in range(self.firstContextIndex, idiomIndexes[0]) ]
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
            contextBeforeIndexes = [ x for x in range(self.firstContextIndex, idiomIndexes[0]) ]
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
        
    def allIdiomAndContextAttns(self, data):
        """
        Return list of average attention to idiom and list of average attention to context
        """
        idiomAttention = []
        wordAttention = []
        for layer in range(12):
            print("layer", layer)
            tokenToTokenAttn = normalizedTokenToToken(attn_list)
            idiomAttentionOneLayer = []
            wordAttentionOneLayer = []
            for i, d in enumerate(data): 
                sampleIdiomAttention, sampleWordAttention = getAvgAttentionsByLayer(d, layer)
                idiomAttentionOneLayer.append(sampleIdiomAttention)
                wordAttentionOneLayer.append(sampleWordAttention)
        return idiomAttention, wordAttention
    
    def attnMeanDiff(self):
        contextToPIEDiff = np.nanmean(self.allIdiomContextToPIE) - np.nanmean(self.allLiteralContextToPIE)
        PIEToPIEDiff = np.nanmean(self.allIdiomPIEToPIE) - np.nanmean(self.allLiteralPIEToPIE)
        PIEToContextDiff = np.nanmean(self.allIdiomPIEtoContext) - np.nanmean(self.allLiteralPIEtoContext)
        #print(PIEToContextDiff, self.allIdiomPIEtoContext, self.allLiteralPIEtoContext)
        entDiff = np.nanmean(self.allIdiomEnt) - np.nanmean(self.allLiteralEnt)
        entToDiff = np.nanmean(self.allIdiomEntTo) - np.nanmean(self.allLiteralEntTo)
        return contextToPIEDiff, PIEToPIEDiff, PIEToContextDiff, entDiff, entToDiff

# %% [original cell 8]
idiomContextToPIE = []
literalContextToPIE = []
idiomPIEToPIE = []
literalPIEToPIE = []
idiomPIEtoContext = []
literalPIEtoContext = []
idiomEnt = []
literalEnt = []
idiomEntTo = []
literalEntTo = []
idiomNorms = []
literalNorms = []
idiomCS = []
literalCS = []

for i in range(12):
    print("layer", i+1)
    attn = Attention(idiom_data, literal_data, i, firstContextIndex=0)
    idiomContextToPIE.append(attn.allIdiomContextToPIE)
    literalContextToPIE.append(attn.allLiteralContextToPIE)
    idiomPIEToPIE.append(attn.allIdiomPIEToPIE)
    literalPIEToPIE.append(attn.allLiteralPIEToPIE)
    idiomPIEtoContext.append(attn.allIdiomPIEtoContext)
    literalPIEtoContext.append(attn.allLiteralPIEtoContext)
    idiomEnt.append(attn.allIdiomEnt)
    literalEnt.append(attn.allLiteralEnt)
    idiomEntTo.append(attn.allIdiomEntTo)
    literalEntTo.append(attn.allLiteralEntTo)
    idiomNorms.append(attn.allIdiomNorms)
    literalNorms.append(attn.allLiteralNorms)
    idiomCS.append(attn.allIdiomCS)
    literalCS.append(attn.allLiteralCS)

# %% [original cell 9]
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
        attn = Attention(idiom_data, literal_data, layer, head=head, firstContextIndex=0)
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

# %% [original cell 10]
def plotHeatmap(data, title='', y_lab='Attention Head'):
    data = np.array(data).T
    x_ticks = [i for i in range(1, 13)]
    y_ticks = [i for i in range(1, 13)]
    bound = np.max(np.abs(data))
    
    s = sns.heatmap(data, xticklabels=x_ticks, yticklabels=y_ticks, cmap='coolwarm', vmin=-bound, vmax=bound)
    s.set(title=title, xlabel='Layer', ylabel=y_lab)
    plt.show()

#  Excluding the First Token
plotHeatmap(PIEToPIE, 'Mean Difference in PIE to PIE Attention')
plotHeatmap(PIEtoContext, 'Mean Difference in PIE to Context Attention')
plotHeatmap(contextToPIE, 'Mean Difference in Context to PIE Attention')
plotHeatmap(ent, 'Mean Difference in Attention Entropy from PIE')
plotHeatmap(entTo, 'Mean Difference in Attention Entropy To PIE')

# %% [original cell 11]
def plotBoxplots(data1, data2, ylimlower=0, ylimupper=0.005, title='', y_lab='Values'):
    df1 = pd.DataFrame(data1).T
    df2 = pd.DataFrame(data2).T
    df1['Type'] = 'Figurative'
    df2['Type'] = 'Literal'
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

# %% [original cell 12]
plotBoxplots(idiomPIEtoContext, literalPIEtoContext, ylimlower=0.56, ylimupper=1.03, 
             title='Mean PIE to Context Attentions', y_lab='Adjusted Attention')
plotBoxplots(idiomPIEToPIE, literalPIEToPIE, ylimlower=0, ylimupper=16.8, 
             title='Mean PIE to PIE Attentions', 
            y_lab='Adjusted Attention')
plotBoxplots(idiomContextToPIE, literalContextToPIE, ylimlower=0.03, ylimupper=1.1, 
             title='Mean Context to PIE Attentions', y_lab='Adjusted Attention')
plotBoxplots(idiomEnt, literalEnt, ylimlower=0.7, ylimupper=3.8, 
             title='Mean Attention Entropy From PIE', 
            y_lab='Attention Entropy')
plotBoxplots(idiomEntTo, literalEntTo, ylimlower=-0.04, ylimupper=0.37, 
             title='Mean Attention Entropy To PIE', 
            y_lab='Attention Entropy')
plotBoxplots(np.log(idiomNorms), np.log(literalNorms), ylimlower=3.75, ylimupper=5.85, 
             title='Log of Embedding Norm of PIE', y_lab='Natural Log of Embedding Norm')
plotBoxplots(idiomCS, literalCS, ylimlower=0.44, ylimupper=1, title='Mean Cosine Similarity within PIE', 
            y_lab='Cosine Similarity')

# %% [original cell 13]
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

# %% [original cell 14]
def plotBoxplots2(data1, data2, ylimlower=0, ylimupper=0.005, title='', y_lab='Cosine Similarity'):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df1['Type'] = 'Figurative'
    df2['Type'] = 'Literal'
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
    
plotBoxplots2(idiomCS, literalCS, ylimlower=0.914, ylimupper=1, 
              title='Mean Cosine Sim of PIE to Previous Layer')

# %% [original cell 15]
def plotViolinPlots(data1, data2, ylimlower=0, ylimupper=0.005, title='', y_lab='Value'):
    plt.figure(figsize=(7, 5))
    df1 = pd.DataFrame(data1).T
    df2 = pd.DataFrame(data2).T
    df1['Type'] = 'Figurative'
    df2['Type'] = 'Literal'
    df = pd.concat([df1, df2], axis=0)
    df['index'] = df.index
    data = df.melt(id_vars=['index', 'Type'], var_name='Layer', value_name='Values')
    ax = sns.violinplot(x="Layer", y="Values", hue="Type", data=data, cut=0, inner='quartile')  # RUN PLOT   
    ax.set_ylim(ylimlower, ylimupper)
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.clf()
    plt.close()
