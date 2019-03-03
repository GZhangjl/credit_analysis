
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# 进行数据导入

companies_data = pd.read_excel(io='companies.xls', sheet_name=0)


# In[3]:


companies_data.head(10)


# In[4]:


# 根据银监会贷款风险分级，RATING共分为5级，现将标记为1、2的企业认定为非违约级，3、4、5认定为违约级
# 进行标签和特征值分离

companies_data['label'] = 1
companies_data['label'][companies_data['RATING']==1] = 0
companies_data['label'][companies_data['RATING']==2] = 0
data_label = companies_data['label']
data_features = companies_data.drop(labels=['RATING', 'label'], axis=1)


# In[5]:


# 进行探索性分析

fig, axe = plt.subplots(nrows=3, ncols=5)
i = 0
for col_name in data_features.columns:
    data_features.boxplot(column=col_name, ax=axe[i//5][i%5], figsize=(10,10))
    i += 1 

# 可以看到极端值比较多，但是由于这已经是有效数据，所以不存在由于收集等因素导致的不正常极端值，故只做观察不做处理


# In[6]:


sns.set_style('whitegrid')
sns.heatmap(data=companies_data.corr(), cmap=sns.color_palette('RdBu', n_colors=200)) 

# 可以得到虽然标签和其他特征无显著关系，但是一些特征之间存在较强相关性，可以减少特征以降维


# In[7]:


# 其他极值、均值等描述性统计量因为与研究问题相关性不强，暂时不做观察

# 进行指标的建立
# 由于都是财务报表中的指标，由于公司之间体量差别较大，所以单看绝对值无法获得有效结果，故查看相关财务比率

data = pd.DataFrame(columns=['x%s' % i for i in range(1,19)])
data['x1'] = data_features['LIBITY'] / data_features['ASSET'] # 总负债/总资产
data['x2'] = data_features['SREVENUE'] / data_features['INTEREST'] # 销售收入/利息费用
data['x3'] = data_features['CASSET'] / data_features['CLIBITY'] # 流动资产/流动负债
data['x4'] = data_features['NPROFIT'] / (data_features['ASSET'] - data_features['LIBITY']) # 净利润/净资产
data['x5'] = data_features['SREVENUE'] / data_features['CASH'] # 销售收入/现金
data['x6'] = np.log(data_features['ASSET']) # 总资产的对数
data['x7'] = data_features['SREVENUE'] / data_features['ASSET'] # 销售收入/总资产
data['x8'] = data_features['SPROFIT'] / data_features['ASSET'] # 销售利润/总资产
data['x9'] = (data_features['SREVENUE'] - data_features['SPROFIT']) / data_features['SPROFIT'] # 销售成本/销售收入
data['x10'] = (data_features['RECEIVAB'] + data_features['INVENTRY']) / (data_features['ASSET'] - data_features['LIBITY']) # (应收账款+存货)/净资产
data['x11'] = data_features['INVENTRY'] / (data_features['ASSET'] - data_features['LIBITY']) # 存货/净资产
data['x12'] = data_features['SREVENUE'] / data_features['LIBITY'] # 销售收入/总负债
data['x13'] = data_features['CASSET'] / (data_features['ASSET'] - data_features['LIBITY']) # 流动资产/净资产
data['x14'] = data_features['SPROFIT'] / data_features['INTEREST'] # 销售利润/利息费用
data['x15'] = data_features['SREVENUE'] / data_features['CASSET'] # 销售收入/流动资产
data['x16'] = data_features['SREVENUE'] / (data_features['ASSET'] - data_features['LIBITY']) # 销售收入/净资产
data['x17'] = data_features['CASSET'] / data_features['INTEREST'] # 流动资产/利息费用
data['x18'] = data_features['MAINCOST'] / data_features['SREVENUE'] # 主营业务成本/销售收入
y = data_label


# In[8]:


data.head(10)


# In[9]:


# 进行训练数据集和验证数据集的拆分

from sklearn.model_selection import train_test_split

train_temp, test_temp, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=13)
X_train = train_temp.copy()
X_test = test_temp.copy()
train = train_temp.copy()
test = test_temp.copy()
train['label'] = y_train
test['label'] = y_test


# In[10]:


# 对训练集进行分析
# 降维
# 1.采用相关系数进行判断

sns.set_style('whitegrid')
sns.heatmap(data=train.corr(), cmap=sns.color_palette('RdBu', n_colors=200)) # 可以得到有一些特征与标签结果相关性小，所以可以剔除以降维

train_corr = train.corr()
train_corr1 = train_corr[train_corr['label'] <= 0.25][train_corr['label'] >= -0.25]
col_keep = train_corr1.index.tolist()
X_train_keep = X_train[col_keep]
X_test_keep = X_test[col_keep]


# In[11]:


# 2.采用LDA进行分析
# （暂不采用PCA（主成分分析法）进行分析，因为PCA适合非监督学习）
# from sklearn.decomposition import PCA
# pca_test = PCA(n_components=data.shape[1])
# pca_test.fit(X_train)
# a = pca_test.explained_variance_ratio_
# plt.plot([i for i in range(data.shape[1])], [np.sum(a[:i+1]) for i in range(data.shape[1])])

# 由于LDA本身就可以作为分类器使用，所以在降维的同时直接也产生了分类模型

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=data.shape[1])
lda.fit(X_train, y_train)
lda_score = lda.score(X_test, y_test)
lda_c_score = lda.decision_function(X_test)


# In[12]:


lda_score


# In[13]:


# 另外可使用混淆矩阵以及相关可视化进行模型评估

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score

lda_y_pred = lda.predict(X_test)
lda_c_m = confusion_matrix(y_test, lda_y_pred)
lda_p_score = precision_score(y_test, lda_y_pred)
lda_r_score = recall_score(y_test, lda_y_pred)
lda_f1_score = f1_score(y_test, lda_y_pred)
lda_p, lda_r, lda_th = precision_recall_curve(y_test, lda_c_score)

# 精确率、召回率曲线

fig, axe = plt.subplots(3, 1)
axe[0].plot(lda_th, lda_p[:-1])
axe[1].plot(lda_th, lda_r[:-1])
axe[2].plot(lda_p, lda_r)


# In[14]:


# ROC曲线

lda_fpr, lda_tpr, lda_th = roc_curve(y_test, lda_c_score)
plt.plot(lda_fpr, lda_tpr)
lda_res = roc_auc_score(y_test, lda_c_score)


# In[15]:


lda_res


# In[16]:


# 使用逻辑回归进行分类以及kNN方法进行分类
# 1.逻辑回归+评估(使用全部特征)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_score = lr.score(X_test, y_test)
lr_c_score = lr.decision_function(X_test)

lr_c_m = confusion_matrix(y_test, lr_y_pred)
lr_p_score = precision_score(y_test, lr_y_pred)
lr_r_score = recall_score(y_test, lr_y_pred)
lr_f1_score = f1_score(y_test, lr_y_pred)
lr_p, lr_r, lr_th = precision_recall_curve(y_test, lr_c_score)

# 精确率、召回率曲线

fig, axe = plt.subplots(3, 1)
axe[0].plot(lr_th, lr_p[:-1])
axe[1].plot(lr_th, lr_r[:-1])
axe[2].plot(lr_p, lr_r)


# In[17]:


# ROC曲线

lr_fpr, lr_tpr, lr_th = roc_curve(y_test, lr_c_score)
plt.plot(lr_fpr, lr_tpr)
lr_res = roc_auc_score(y_test, lr_c_score)


# In[18]:


lr_res


# In[19]:


# (使用降维后的数据)

from sklearn.linear_model import LogisticRegression
lr2 = LogisticRegression()
lr2.fit(X_train_keep, y_train)
lr2_y_pred = lr2.predict(X_test_keep)
lr2_score = lr2.score(X_test_keep, y_test)
lr2_c_score = lr2.decision_function(X_test_keep)

lr2_c_m = confusion_matrix(y_test, lr2_y_pred)
lr2_p_score = precision_score(y_test, lr2_y_pred)
lr2_r_score = recall_score(y_test, lr2_y_pred)
lr2_f1_score = f1_score(y_test, lr2_y_pred)
lr2_p, lr2_r, lr2_th = precision_recall_curve(y_test, lr2_c_score)
fig, axe = plt.subplots(3, 1)
axe[0].plot(lr2_th, lr2_p[:-1])
axe[1].plot(lr2_th, lr2_r[:-1])
axe[2].plot(lr2_p, lr2_r)


# In[20]:


lr2_fpr, lr2_tpr, lr2_th = roc_curve(y_test, lr2_c_score)
plt.plot(lr_fpr, lr_tpr)
lr2_res = roc_auc_score(y_test, lr2_c_score)


# In[21]:


lr2_res

# 不理想


# In[22]:


# 2.kNN+评估
# (使用网格搜索和交叉验证)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
grid_param = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 20)]},
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 20)],
        "p": [i for i in range(1, 10)]
    }
]
gridcv_knn = GridSearchCV(knn, grid_param, n_jobs=-1, verbose=10)
gridcv_knn.fit(X_train, y_train)
best_est = gridcv_knn.best_estimator_
beat_pred = best_est.predict(X_test)
best_score_model = gridcv_knn.best_score_
knn_cof = confusion_matrix(y_test, beat_pred)
best_param = gridcv_knn.best_params_
best_c_m = confusion_matrix(y_test, beat_pred)


# In[23]:


# 最好的模型

best_est


# In[24]:


# 最好的模型超参

best_param


# In[25]:


# 最好的模型得分

best_score_model


# In[26]:


# 对应的混淆举证

best_c_m

