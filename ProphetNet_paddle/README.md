# ProphetNet-paddle
将原文torch的ProphetNet通过Paddle深度学习框架实现。

## Dependency

- pip install paddlepaddle
- pip install paddlenlp  

## 模型对齐

模型文件存于prophetnet目录中。包含ngram_multihead_attention_paddle.py和ngram_s2s_model_paddle.py
前向对齐精度约为10-5级别。
## 模型权重对齐

基于paddle的模型与原文torch模型一一对齐，paddle模型state_dict参数key与原torch模型state_dict参数key相同。 
由torch2paddle_weight.py实现。转换后的权重存于pretrained_checkpoints/prophetnet-large-uncased.pdparams、pretrained_checkpoints/gigaword/finetune_gigaword_checkpoints/prophetnet-large-uncased-squad-qg.pdparams和cnndm/finetune_cnndm_checkpoints/prophetnet-large-uncased-cnndm.pdparams。


## 数据预处理
将原语料文件转换为tokenized文件。对于CNN/DailyMail，通过cnndm/preprocess_cnn_dm.py处理，将cnndm/original_data中的文件，数据预处理存于cnndm/prophetnet_tokenized。
对于Gigaword，通过gigaword/preprocess_gigaword.py处理，将gigaword/original_data中的文件，数据预处理存于gigaword/prophetnet_tokenized。

## 微调
train.py包含数据集加载、ID转换、创建数据集迭代器、加载预训练参数、设置优化器参数、训练过程。（尚未实现训练对齐）