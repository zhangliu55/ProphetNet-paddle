import tqdm
import numpy as np
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
import paddle
from functools import partial
from paddlenlp.transformers import BertTokenizer
import time
import paddle

#加载tokenized数据集
def read(data_src,data_tgt, max_len=512):
    data_src = open(data_src, 'r', encoding='utf-8')
    tokens_list_src=[]
    for line_src in tqdm.tqdm(data_src.readlines()):
        tokens_src = line_src.strip().split(' ')[:max_len-1]
        tokens_list_src.append(tokens_src)

    data_tgt = open(data_tgt, 'r', encoding='utf-8')
    tokens_list_tgt=[]
    for line_tgt in tqdm.tqdm(data_tgt.readlines()):
        tokens_tgt = line_tgt.strip().split(' ')[:max_len-1]
        tokens_list_tgt.append(tokens_tgt)

    for i in range(len(tokens_list_tgt)):
        yield {'tokens_src': tokens_list_src[i], 'tokens_tgt': tokens_list_tgt[i]}

train_dataset= load_dataset(read, data_src='prophetNet/tokenized/cnndm_tokenized/train.src',
                   data_tgt='prophetNet/tokenized/cnndm_tokenized/train.tgt',
                   lazy=False)
# test_dataset= load_dataset(read, data_src='prophetNet/cnndm/prophetnet_tokenized/test.src',
#                    data_tgt='prophetNet/cnndm/prophetnet_tokenized/test.tgt',
#                    lazy=False)


#tokenized转换为ID
bpe = BertTokenizer.from_pretrained('bert-base-uncased')
bpe.vocab = bpe.load_vocabulary(filepath="vocab1.txt",
                                unk_token="[UNK]", pad_token="[PAD]", bos_token="[CLS]", eos_token="[SEP]")

def convert_example(example, tokenizer=bpe):
    tokens_id_src = example["tokens_src"] + ["[SEP]"]
    tokens_id_src = bpe.convert_tokens_to_ids(tokens_id_src)
    src_length = len(tokens_id_src)
    tokens_id_prev = ["[SEP]"] + example["tokens_tgt"]
    tokens_id_prev = bpe.convert_tokens_to_ids(tokens_id_prev)

    tokens_id_tgt = example["tokens_tgt"] + ["[SEP]"]
    tokens_id_tgt = bpe.convert_tokens_to_ids(tokens_id_tgt)

    return tokens_id_src, src_length, tokens_id_prev, tokens_id_tgt

train_dataset = train_dataset.map(partial(convert_example, tokenizer=bpe))
# test_dataset = test_dataset.map(partial(convert_example, tokenizer=bpe))

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
def create_dataloader(dataset,mode='train',batch_size=1,batchify_fn=None):

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=bpe.pad_token_id,pad_right=False),  # tokens_id_src
    Stack(),  # src_length
    Pad(axis=0, pad_val=bpe.pad_token_id),  # tokens_id_prev
    Pad(axis=0, pad_val=bpe.pad_token_id) # tokens_id_tgt
): [data for data in fn(samples)]


#加载预训练模型
from  prophetnet.ngram_s2s_model_paddle import NgramTransformerProphetModel as NgramTransformerProphetModel_paddle
model_paddle = NgramTransformerProphetModel_paddle(
  src_vocab_size=30522,
  trg_vocab_size=30522,
  max_length=512,
  num_encoder_layers=12,
  num_decoder_layers=12,
  emb_dim=1024,
  dropout=0.1,
  bos_id=0,
  eos_id=1)

params_path="pretrained_checkpoints/prophetnet-large-uncased.pdparams"
model_paddle.load_pretrainmodel(params_path)

# 训练轮次
epochs = 10

# 创建训练集迭代器
train_data_loader = create_dataloader(train_dataset, mode='train', batch_size=4, batchify_fn=batchify_fn)

# AdamW优化器
scheduler = paddle.optimizer.lr.NoamDecay(d_model=512, warmup_steps=1000, learning_rate=1e-7, last_epoch=0)
clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
optimizer = paddle.optimizer.AdamW(
    learning_rate=scheduler,
    beta1=0.9,
    beta2=0.999,
    grad_clip=clip,
    parameters=model_paddle.parameters())

# 开启训练
global_step = 0
model_paddle.train()
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        tokens_id_srcs, src_lengths, tokens_id_prevs, tokens_id_tgts = batch
        loss, sample_size = model_paddle.NgramLmLoss(src_tokens=tokens_id_srcs,
                                                     src_lengths=src_lengths,
                                                     prev_output_tokens=tokens_id_prevs,
                                                     targets=tokens_id_tgts)
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        global_step += 1
        if global_step % 10 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s" % (
            global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()
