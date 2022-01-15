# 比赛思路Highlights

## 任务背景

**任务**：预测节点(论文)类别。

**数据**：基于微软学术文献生成的论文关系图，其中的节点是论文，边是论文间的引用关系。包括约**150**万个节点，**2000**万条边。节点包含300维的特征来自论文的标题和摘要等内容。

**关键词**：图神经网络、节点分类、单分多类

[比赛地址](https://www.biendata.xyz/competition/maxp_dgl/)，对比赛中优秀解决方案进行复盘整理。

[比赛答辩记录](https://www.bilibili.com/video/BV1fr4y1v737?p=2)

## Rank 1

### 代码实现（暂无）

### 数据处理

1. 验证集划分：考虑到训练集和测试集节点度上的差异，因此根据测试集出入度信息从训练集中采样相同分布的数据作为验证集
2. 训练时采样策略：random + top k采样（优先采样有标签的邻居）
3. 利用神经网络的记忆性，将训练集分为head和tail两个训练子集，且tail训练子集的分布和测试集分布尽可能一致

### 模型

1. 分治：孤立节点用MLP，非孤立节点用GNN

2. GAT+NormAdj作为base模型，几种变种：加入门控残差、加入FFN层、加入degree-aware残差层

   $\boldsymbol{h}^{(l+1)}=\sigma\left(D^{-1 / 2} A_{a t t} D^{-1 / 2} \boldsymbol{h}^{l} \boldsymbol{W}\right)$，其中$A_{a t t}$是得到的注意力矩阵（在答辩时评委提问较多）

3. 标签进行label smoothing，统计每种标签的邻居标签分布的关系矩阵，进行特征的平滑，$\tilde{y}=M^{T} y \times s+y \times(1-s)$,其中$y$是one-hot标签，利用新的标签计算交叉熵损失$\operatorname{loss}=-\sum \tilde{y}^{T} \log (p)$

   

### 集成&后处理

1. MLP模型通过调整不同超参生成多模型
2. GNN进行两轮训练：第一轮使用原始数据训练，第二轮加入微标签数据，两轮训练结果+多GNN变种进行融合

### 训练时间

单模型1个epoch需要约1个小时

### 疑问

1. GAT+NormAdj有效的机理
2. 在进行label平滑的时候可能会对原有特征有一定程度弱化，为什么会有效

## Rank 2

### [代码实现](https://github.com/langgege-cqu/maxp_dgl)

### 数据处理

1. 网络嵌入特征Deepwalk + 统计特征（节点出入度以及一二阶邻域信息）

### 模型

1.  GCN + graph SAGE(LSTM聚合) + GAT + GATv2Conv，逐层融合不同基础图网络特征
2. 两种特征融合方式：基于多头注意力机制融合，基于SENet融合

### 训练细节

drop edge, masked label prediction, k-folds validation

### 其他尝试

- 考虑改善类与类连结边缘节点的预测准确率，引入三个通道并融合
- LAGNN：使用CVAE模型
- 损失函数改进
- 构建引用、被引用关系的异构图
- 伪标签增强
- 模型自蒸馏，对比学习....

## Rank 3

### 代码实现

提供了一个[baseline](https://github.com/minghaochen/2021-MAXP-DGL-GraphML-competition-intermediate-solution-)思路

### 数据处理

1. 数据观察：标签不平衡、孤立节点
2. 没有引用关系但内容相近的论文也可能属于同一类，构造特征相似图
3. deepwalk特征，node2vec特征

### 模型

1. 没有引用关系但内容相近的论文也可能属于同一类，构造特征相似图
2. GAT变种（也利用了和rank 1类似的NormAdj）
3. 同构图模型+异构图模型（引用、被引用、self- loop，孤立节点的特征近邻图）
4. Robust loss function

### 模型集成

k folds + 同构图异构图模型融合

### 其他尝试

- 统计特征传播会泄露
- 训练集和验证集统计特征分布不一致（参考rank 1）
- 自蒸馏、伪标签、correct&smooth

## Rank 4

### 代码实现（暂无）

### 数据处理

1. PCA降维
2. 统计特征：出入度、pagerank分数、节点hits分数
3. Deepwalk特征
4. 数据增强
   1. 两次梯度下降、第一次进行扰动、加入对抗样本后重新训练
   2. 邻居label标签中一半进行mask

### 模型

1. 属性特征利用SAGE聚合，标签特征用GAT聚合
2. 构造异构图（cite, cited, self-loop）版本的模型
3. 学习Entropy minimization，增大分类的类间距离；学习节点属性映射

### 其他尝试

- Link prediction有用，可以进行任务补充
- 观察到测试集节点很少被引用，在训练时直接去掉被引用边关系（有效果）

## Rank 5

### 代码实现（暂无）

除原始特征外加入了深度游走特征，在模型层面上使用ResGAT，推理时多次小尺度邻居采样最后进行投票融合。

## Rank 6

### [代码实现](https://github.com/ytchx1999/MAXP_DGL_Graph)

进行了GAMLP的预处理：邻居聚合k次，生成k-hop邻居的节点特征列表。base模型是GAMLP（和前排方法差别较大），训练快。进行了C&S操作

### 疑问

GAMLP面对图的边上有权重或者异构图不适用

## 其他

[Rank 7 solution](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M)

[官方baseline](https://github.com/dglai/maxp_baseline_model)


