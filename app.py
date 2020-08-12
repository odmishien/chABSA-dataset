#!/usr/bin/env python
# coding: utf-8

# # 1.chABSAデータセットを読み込み、DataLoaderの作成(BertのTokenizerを利用）

# In[1]:


# パスの追加（必要に応じて）
import sys
sys.path.append('/home/siny/miniconda3/envs/pytorch/lib/python36.zip')
sys.path.append('/home/siny/miniconda3/envs/pytorch/lib/python3.6')
sys.path.append('/home/siny/miniconda3/envs/pytorch/lib/python3.6/lib-dynload')
sys.path.append('/home/siny/.local/lib/python3.6/site-packages')
sys.path.append('/home/siny/miniconda3/envs/pytorch/lib/python3.6/site-packages')


# In[11]:


import random
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext


# In[12]:


from utils.dataloader import get_chABSA_DataLoaders_and_TEXT
from utils.bert import BertTokenizer


# In[13]:


train_dl, val_dl, TEXT, dataloaders_dict= get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32)


# In[14]:


# 動作確認 検証データのデータセットで確認
batch = next(iter(train_dl))
print("Textの形状=", batch.Text[0].shape)
print("Labelの形状=", batch.Label.shape)
print(batch.Text)
print(batch.Label)


# In[15]:


# ミニバッチの1文目を確認してみる
tokenizer_bert = BertTokenizer(vocab_file="./vocab/vocab.txt", do_lower_case=False)
text_minibatch_1 = (batch.Text[0][1]).numpy()

# IDを単語に戻す
text = tokenizer_bert.convert_ids_to_tokens(text_minibatch_1)

print(text)


# # 2.BERTによるネガポジ分類モデル実装

# In[16]:


from utils.bert import get_config, BertModel,BertForchABSA, set_learned_params

# モデル設定のJOSNファイルをオブジェクト変数として読み込みます
config = get_config(file_path="./weights/bert_config.json")

# BERTモデルを作成します
net_bert = BertModel(config)

# BERTモデルに学習済みパラメータセットします
net_bert = set_learned_params(
    net_bert, weights_path="./weights/pytorch_model.bin")


# In[17]:


# モデル構築
net = BertForchABSA(net_bert)

# 訓練モードに設定
net.train()

print('ネットワーク設定完了')


# # 3.BERTのファインチューニングに向けた設定

# In[18]:


# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. まず全部を、勾配計算Falseにしてしまう
for name, param in net.named_parameters():
    param.requires_grad = False

# 2. 最後のBertLayerモジュールを勾配計算ありに変更
for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

# 3. 識別器を勾配計算ありに変更
for name, param in net.cls.named_parameters():
    param.requires_grad = True


# In[19]:


# 最適化手法の設定

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 5e-5}
], betas=(0.9, 0.999))

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算


# In[ ]:


# 学習・検証を実施
from utils.train import train_model

# 学習・検証を実行する。
num_epochs = 1
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)


# In[ ]:


# 学習したネットワークパラメータを保存します
save_path = './weights/bert_fine_tuning_chABSA_test.pth'
torch.save(net_trained.state_dict(), save_path)


# # 4.サンプルの文章で推論とAttentionを可視化する。

# In[20]:


from config import *
from predict import predict, create_vocab_text, build_bert_model
from IPython.display import HTML, display


# In[21]:


#TEXTオブジェクト（torchtext.data.field.Field）をpklファイルにダンプしておく（推論時に利用するため）
# 1度生成すればＯＫ
TEXT = create_vocab_text()


# In[22]:


# 学習モデルのロード
net_trained = BertForchABSA(net_bert)
save_path = './weights/bert_fine_tuning_chABSA_22epoch_1123.pth'   #学習済みモデルを指定
# 学習したネットワークパラメータをロード
net_trained.load_state_dict(torch.load(save_path, map_location='cpu'))
net_trained.eval()


# In[30]:


input_text = "損益面におきましては、経常収益は、貸出金利息や有価証券売却益の減少により、前期比72億73百万円減少の674億13百万円となりました"
#net_trained = build_bert_model()
net_trained.eval()
html_output = predict(input_text, net_trained)
print("======================推論結果の表示======================")
print(input_text)
display(HTML(html_output))


# # 5.テストデータで一括予測する。

# In[31]:


from utils.config import *
from utils.predict import predict2, create_vocab_text, build_bert_model
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[39]:


df = pd.read_csv("test.csv", engine="python", encoding="utf-8-sig")
df["PREDICT"] = np.nan   #予測列を追加
net_trained.eval()  #推論モードに。

for index, row in df.iterrows():
    df.at[index, "PREDICT"] = predict2(row['INPUT'], net_trained).numpy()[0]  # GPU環境の場合は「.cpu().numpy()」としてください。
    
df.to_csv("predicted_test .csv", encoding="utf-8-sig", index=False)


# ## 混合行列を表示

# In[25]:


#混合行列の表示（評価）

y_true =[]
y_pred =[]
df = pd.read_csv("predicted_test .csv", engine="python", encoding="utf-8-sig")
for index, row in df.iterrows():
    if row['LABEL'] == 0:
        y_true.append("negative")
    if row['LABEL'] ==1:
        y_true.append("positive")
    if row['PREDICT'] ==0:
        y_pred.append("negative")
    if row['PREDICT'] ==1:
        y_pred.append("positive")

    
print(len(y_true))
print(len(y_pred))


# 混同行列(confusion matrix)の取得
labels = ["negative", "positive"]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# データフレームに変換
cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)

# 結果の表示
cm_labeled


# In[26]:



y_true =[]
y_pred =[]
df = pd.read_csv("predicted_test .csv", engine="python", encoding="utf-8-sig")
for index, row in df.iterrows():
    y_true.append(row["LABEL"])
    y_pred.append(row["PREDICT"])
        
print("正解率（すべてのサンプルのうち正解したサンプルの割合）={}%".format((round(accuracy_score(y_true, y_pred),2)) *100 ))
print("適合率（positiveと予測された中で実際にpositiveだった確率）={}%".format((round(precision_score(y_true, y_pred),2)) *100 ))
print("再現率（positiveなデータに対してpositiveと予測された確率）={}%".format((round(recall_score(y_true, y_pred),2)) *100 ))
print("F1（適合率と再現率の調和平均）={}%".format((round(f1_score(y_true, y_pred),2)) *100 ))



# In[ ]:




