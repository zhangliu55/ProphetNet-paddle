# ProphetNet-paddle
他提出了一种新的序列到序列的预训练模型，称为ProphetNet，该模型引入了一种新的自我监督目标，称为未来n-gram预测，并提出了n流自我注意机制。his paper presents a new sequence-to- sequence pre-training model called ProphetNet, which introduces a novel self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. 
# ProphetNet-torch原文代码复现
根据百度论文复现要求，在./ProphetNet_torch中对cnnDM和Gigaword数据集的测试集rouge指标进行原文代码复现，复现结果指标分别存于/ProphetNet_torch/cnndm/score_ck9_pelt1.2_test_beam5.txt和/ProphetNet_torch/gigaword/score_ck7_pelt1.0_test_beam4.txt中。  
cnnDM:ROUGE-F(1/2/l): 44.20/21.17/41.30  
Gigaword:ROUGE-F(1/2/l): 39.51/20.42/36.69  
与原文中指标一致。

