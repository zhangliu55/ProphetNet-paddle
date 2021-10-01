import paddle
from paddlenlp.transformers import ErnieModel, ErnieTokenizer, BertTokenizer
from paddlenlp.transformers import *
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import math
from ngram_multihead_attention_paddle import NgramMultiheadAttention, ngram_attention_bias

from scipy.io import loadmat,savemat


DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512

class NgramTransformerProphetModel(nn.Layer):
    def __init__(self,
          src_vocab_size,
          trg_vocab_size,
          max_length,
          num_encoder_layers=12,
          num_decoder_layers=12,
          # n_head=,
          emb_dim=1024,
          # d_inner_hid,
          dropout=0.1,
          share_all_embeddings=True,
          attn_dropout=None,
          act_dropout=None,
          bos_id=0,
          eos_id=1,
          max_source_positions=512,
          max_target_positions=512,

               ):
        super(NgramTransformerProphetModel,self).__init__()

        #########################paddle#########################
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = emb_dim
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout
        if max_source_positions is None:
            max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
        if max_target_positions is None:
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS

        if share_all_embeddings:
            encoder_embed_tokens=nn.Embedding(
                num_embeddings=src_vocab_size,
                embedding_dim=emb_dim,
                # padding_idx=bos_id,
                weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0.,emb_dim**-0.5)))
            decoder_embed_tokens = encoder_embed_tokens
        else:
            encoder_embed_tokens = nn.Embedding(
                num_embeddings=src_vocab_size,
                embedding_dim=emb_dim,
                # padding_idx=bos_id,
                weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0.,emb_dim**-0.5)))
            decoder_embed_tokens = nn.Embedding(
                num_embeddings=src_vocab_size,
                embedding_dim=emb_dim,
                # padding_idx=bos_id,
                weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0.,emb_dim**-0.5)))
        np.random.seed(666)
        w0 = np.random.rand(30522, 1024).astype(np.float32)
        encoder_embed_tokens.weight.set_value(w0)
        # print(encoder_embed_tokens.state_dict().keys())
        self.encoder = TransformerEncoder(dropout=dropout,
                          max_source_positions=max_source_positions,
                          embed_tokens=encoder_embed_tokens,
                          encoder_ffn_embed_dim=4096,
                          encoder_attention_heads=16,
                          attention_dropout=0.1,
                          activation_dropout=0.1,
                          activation_fn="gelu",
                          num_layers=12
                          )
        self.decoder = NgramTransformerDecoder(ngram=2,
                            num_buckets=32,
                            embed_tokens=decoder_embed_tokens,
                            relative_max_distance=128,
                            dropout=dropout,
                            share_decoder_input_output_embed=True,
                            decoder_embed_dim=emb_dim,
                            max_target_positions=max_target_positions,
                            num_layers=12,
                            no_encoder_attn=False
                            )

    def forward(self,src_tokens=None, src_lengths=None, prev_output_tokens=None):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        predict = self.decoder(prev_output_tokens,encoder_out=encoder_out,incremental_state=None)
        return predict
    def encoder_forward(self,src_tokens=None, src_lengths=None):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        return encoder_out
    def decoder_forward(self,prev_output_tokens=None,encoder_out=None):
        predict=self.decoder( prev_output_tokens,encoder_out=encoder_out,incremental_state=None)
        return predict
    def NgramLmLoss(self,src_tokens=None, src_lengths=None, prev_output_tokens=None,targets=None):
        logits_list=self.forward(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=prev_output_tokens)[0]
        print(logits_list.shape)
        ngram = len(logits_list)
        # [B, ngram, T]
        self.padding_idx=0
        expend_targets = paddle.full([ngram,targets.shape[0],targets.shape[1]],self.padding_idx,paddle.int64)
        for i in range(ngram):
            padding_targets = paddle.full_like(targets,self.padding_idx)      
            expend_targets[i,:,:] = targets
        targets =expend_targets
        # print(np.shape(targets))
        logits = paddle.concat(logits_list, axis=0) #.view(ngram, *logits_list[0].size())
        # print(np.shape(logits))
        lprobs = paddle.nn.functional.log_softmax(logits.reshape([-1,logits.shape[-1]]),axis=-1,dtype=paddle.float32)
        # print(np.shape(lprobs))
        loss = paddle.nn.functional.nll_loss(input=lprobs,label=targets.reshape([-1]),reduction='sum',ignore_index=self.padding_idx)

        self.eps=0.1
        if self.eps > 0.:
            smooth_loss = -lprobs.sum(axis=-1, keepdim=True)
            non_pad_mask = targets.not_equal(paddle.to_tensor(self.padding_idx)).reshape([-1])
            smooth_loss=smooth_loss.numpy()
            smooth_loss = paddle.to_tensor(smooth_loss[non_pad_mask])
            smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.shape[-1]
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = paddle.cast(targets.not_equal(paddle.to_tensor(self.padding_idx)),dtype=paddle.int64).sum().numpy()
        #   print("loss:",loss,sample_size)
        return loss,sample_size
    def load_pretrainmodel(self,params_path=None):
        # model = self(
        #   src_vocab_size=30522,
        #   trg_vocab_size=30522,
        #   max_length=512,
        #   num_encoder_layers=12,
        #   num_decoder_layers=12,
        #   # n_head=8,
        #   emb_dim=1024,
        #   # d_inner_hid=2048,
        #   dropout=0.1,
        #   bos_id=0,
        #   eos_id=1)
        state_dict = paddle.load(params_path)
        self.set_dict(state_dict) 
    



class TransformerEncoderLayer(nn.Layer):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',

          ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        #paddle
        # Initialize blocks
        self.activation_fn = nn.GELU()        
        self.self_attn = nn.MultiHeadAttention(
            embed_dim =self.embedding_dim,
            num_heads =num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        # self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim, weight_attr = paddle.ParamAttr(
        #     initializer=paddle.nn.initializer.Assign(value=)))
        # self.fc1.weight.set_value(w0)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        

    def forward(
            self,
            x: paddle.Tensor,
            self_attn_mask: paddle.Tensor = None,
            self_attn_padding_mask: paddle.Tensor = None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """

        residual = x

        # print('encoder_Layer x:',self.dropout,self.training,np.shape(x),self_attn_mask,x)
        x = paddle.transpose(x, perm=[1, 0, 2]) #转置
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask
        )
        x = paddle.transpose(x, perm=[1, 0, 2]) #转置
        # print('encoder_Layer self_attn:',self.dropout,self.training,np.shape(x),x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # print('encoder_Layer self_attn_layer_norm:', self.dropout, self.training, np.shape(x), x)

        residual = x
        x=self.fc1(x)                         #全连接第一层，带激活函数
         
        x = self.activation_fn(x)
        
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)                       #全连接第二层，不带激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x        
        x = self.final_layer_norm(x)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":x.numpy()})         
        # print('encoder_Layer final_layer_norm:', self.dropout, self.training, np.shape(x), x)
        return x

class TransformerEncoder(nn.Layer):
  def __init__(self,
        dropout,
        max_source_positions,
        embed_tokens,
        encoder_ffn_embed_dim=4096,
        encoder_attention_heads=16,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="gelu",
        num_layers=12,
        padding_idx=0
        ):
    super(TransformerEncoder,self).__init__()
    # self.register_buffer('version', torch.Tensor([3]))

    self.dropout =dropout

    embed_dim = embed_tokens._embedding_dim
    # self.padding_idx = embed_tokens._padding_idx
    self.padding_idx = padding_idx
    self.max_source_positions = max_source_positions

    self.embed_tokens = embed_tokens
    self.embed_scale = None #math.sqrt(embed_dim)
    # 位置嵌入
    self.embed_positions = LearnedPositionalEmbedding(
        max_source_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
    )

    self.layers = nn.LayerList([
        TransformerEncoderLayer(
            embed_dim,
            encoder_ffn_embed_dim,
            encoder_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
        )
        for i in range(num_layers)
    ])
    self.emb_layer_norm = nn.LayerNorm(embed_dim)  #paddle

    # self.apply(init_bert_params)

  def forward(self,src_tokens, src_lengths):
    # compute padding mask
    encoder_padding_mask = (src_tokens==self.padding_idx)
    if not encoder_padding_mask.any():
        encoder_padding_mask = None
    x = self.embed_tokens(src_tokens)  #tokens嵌入
    # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
    #   {"x":x.numpy()}) 
    # embed tokens and positions
    if self.embed_scale is not None:
        x *= self.embed_scale
    # print('embed tokens:',np.shape(x),x)
    if self.embed_positions is not None:
        pos_emb, real_positions = self.embed_positions(src_tokens)  #positions嵌入
        x += pos_emb
       
    # print('embed tokens embed_positions:',np.shape(x),x)
    if self.emb_layer_norm:
        x = self.emb_layer_norm(x)  #嵌入层norm
         
    # print('嵌入层norm:',np.shape(x),x)
    x = F.dropout(x, p=self.dropout, training=self.training)  #嵌入层dropout
    if encoder_padding_mask is not None:
        x *= 1 - paddle.cast(paddle.unsqueeze(encoder_padding_mask,-1),'uint8')    
    # print('encoder_padding_mask:',self.dropout,self.training,np.shape(x),x)
    # B x T x C -> T x B x C   
    x = paddle.transpose(x, perm=[1, 0, 2]) #转置
    # print('transpose:',self.dropout,self.training,np.shape(x),x)
    # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
    #   {"x":x.numpy()}) 
   
    for layer in self.layers:
        x = layer(x, self_attn_padding_mask=encoder_padding_mask,)
        # break
    # print('encoder_layer_list:',self.dropout,self.training,np.shape(x),x)
    # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
    #   {"x":x.numpy()})  
    # print(encoder_padding_mask)    
    return {
        'encoder_out': x,  # T x B x C
        'encoder_padding_mask': encoder_padding_mask,  # B x T
    }

class NgramTransformerDecoderLayer(nn.Layer):
    def __init__(
            self,
            ngram=2,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,

    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = nn.GELU()

        self.ngram_self_attn=NgramMultiheadAttention(
                1024,
                16,
                dropout=0.1,
                add_bias_kv=False,
                add_zero_attn=False,
                self_attention=True,
                ngram=2
            )
        self.ngram = ngram
        
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.encoder_attn = nn.MultiHeadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_attention_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=attention_dropout,
            # encoder_decoder_attention=True,
            need_weights=(not self.training and self.need_attn),
        )
 

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.need_attn = False

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            ngram_mask_matrix=None,
            i_buckets_main_stream=None,
            i_bucket_relative_stream=None,
            real_positions=None
    ):
        # one main stream and ngram predicting streams
        residual = x
        # print("x:",np.shape(x),x)
         
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)      
        
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x.numpy()})        
        x, attn = self.ngram_self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            need_weights=False,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions
        )

       
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        x = self.encoder_attn(
            query=x.transpose([1,0,2]),          #[207, 2, 1024]
            key=encoder_out.transpose([1,0,2]),  #[325, 2, 1024]
            value=encoder_out.transpose([1,0,2]), #[325, 2, 1024]
            # attn_mask=encoder_mask,
        ) 
        x=x.transpose([1,0,2]) 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x.numpy()})

        return x, attn

    # def make_generation_fast_(self, need_attn=False, **kwargs):
    #     self.need_attn = need_attn

class NgramTransformerDecoder(nn.Layer):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, ngram,
                 num_buckets,
                 embed_tokens,
                 relative_max_distance,
                 dropout,
                 share_decoder_input_output_embed,
                 decoder_embed_dim,
                 max_target_positions,
                 num_layers=12,
                 padding_idx = 0,
                 no_encoder_attn=False):
        super().__init__()
        # self.register_buffer('version', torch.Tensor([3]))
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance

        self.dropout = dropout
        self.share_input_output_embed = share_decoder_input_output_embed

        input_embed_dim = embed_tokens._embedding_dim
        embed_dim = decoder_embed_dim

        # self.padding_idx = embed_tokens._padding_idx
        self.padding_idx = padding_idx
        self.max_target_positions = max_target_positions
        self.embed_dim = embed_dim
        self.embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.embed_positions = LearnedPositionalEmbedding(
            max_target_positions + 2 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.ngram_input_embed=nn.Embedding(num_embeddings=self.ngram,
                          embedding_dim=input_embed_dim,
                          weight_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Normal(0.,input_embed_dim**-0.5)))
        np.random.seed(666)
        w0=np.random.rand(self.ngram,input_embed_dim).astype(np.float32)
        # w0[0]=0.0
        self.ngram_input_embed.weight.set_value(w0)

        # self.coder_layer=NgramTransformerDecoderLayer(
        #             ngram=2,
        #             embedding_dim=1024,
        #             ffn_embedding_dim=4096,
        #             num_attention_heads=16,
        #             dropout=0.1,
        #             attention_dropout=0.1,
        #             activation_dropout=0.1,
        #             activation_fn="gelu",
        #         )
        # 2 1024 4096 16 0.1 0.1 0.1 gelu
        self.layers = nn.LayerList([
                NgramTransformerDecoderLayer(
                    ngram=2,
                    embedding_dim=1024,
                    ffn_embedding_dim=4096,
                    num_attention_heads=16,
                    dropout=0.1,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    activation_fn="gelu",
                )
            for i in range(num_layers)
        ])

        if not self.share_input_output_embed:
            # self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.embed_dim))
            # nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim ** -0.5)
            self.embed_out = paddle.create_parameter(shape=(len(dictionary),self.embed_dim),
                                  dtype="float32",
                                  default_initializer=nn.initializer.Normal(0.,self.embed_dim**-0.5))
        
        self.emb_layer_norm = nn.LayerNorm(embed_dim)  # paddle
        # self.apply(init_bert_params)

    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                **unused):
        # T
        T = paddle.shape(prev_output_tokens)[1]
        # print("encoder_out:",encoder_out)
        # x [B, (1+ngram)*T, C]
        x_list, extra = self.extract_features(prev_output_tokens,
                            encoder_out,
                            incremental_state, **unused)
        
        x_predicted = x_list[1:]
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x_predicted[0].numpy()})        
        x_predicted = [self.output_layer(x) for x in x_predicted]
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x_predicted[0].numpy()})         
        if incremental_state is not None:
            x_predicted = x_predicted[0]
            for k in extra:
                if extra[k] is not None:
                    extra[k] = extra[k][0]
        return x_predicted, extra
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        # [bos, A, B, C, D, eos] with real positions [1,2,3,4,5,6](main stream), [2,3,4,5,6,7](predicting stream)
        # target [B,C,D] with prev [A,B,C] from [A,B,C,D] as pretraining span with real positions [2,3,4],
        # but target actually [3,4,5] for fine tune with another [bos].
        # thus [2,3,4] used for main stream shifted prev [A,B,C], [3,4,5] used for predicting [B,C,D]
        # print(unused,prev_output_tokens)
        if 'positions' in unused:
            # pretrain procedure
            main_stream_pos_embed = self.embed_positions._forward(unused['positions'])
            real_positions = unused['positions']
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_pretrain_relative_positions(real_positions)
        else:
            # fine tune procedure
            main_stream_pos_embed, real_positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None         #prev_output_tokens 位置嵌入
            if incremental_state is not None:
                i_buckets_main_stream, i_bucket_relative_stream = None, None
            else:
                i_buckets_main_stream, i_bucket_relative_stream = \
                    self.cal_finetune_relative_positions(real_positions)
        #
        # print("real_positions:", real_positions)
 
        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        x = self.embed_tokens(prev_output_tokens)   #prev_output_tokens tokens嵌入
        
        # print("embed_tokens:", x)

        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed             #prev_output_tokens tokens+位置嵌入
 
        # B x T x C -> T x B x C
        # print("x:", x)
        x=paddle.transpose(x,perm=[1,0,2])
        attn = None
        # print("x:",x)

        inner_states = [x]
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight   #ngram_input tokens嵌入
        # print("x:", ngram_input_embed)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":ngram_input_embed.numpy()}) 
        if incremental_state is not None:
            B = paddle.shape(x)[1]
            ngram_masks = [paddle.transpose(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed, perm=[1,0,2])
                for ngram in range(self.ngram)]
            ngram_masks=[paddle.concat([ngram_mask for i in range(B)], axis=1) for ngram_mask in ngram_masks]    
        else:
            ngram_masks = [paddle.transpose(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed, perm=[1,0,2]) for  #ngram_masks     
                           ngram in range(self.ngram)]
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":ngram_masks[0].numpy()})                            
        # print("ngram_masks:",np.shape(ngram_masks),ngram_masks)
        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None            #self_attn_mask  
        # print("self_attn_mask:",self_attn_mask)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":self_attn_mask.numpy()})         
        ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None    #ngram_mask_matrix 
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":ngram_mask_matrix.numpy()})        
        # print("ngram_mask_matrix:",np.shape(ngram_mask_matrix),ngram_mask_matrix)
        # TODO in train [(1+ngram)*T, B, C], in inference [T+ngram, B, C]
        
        x = paddle.concat([x] + ngram_masks, 0)
         
        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x.numpy()})        
        # decoder layers 
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions
            )
            inner_states.append(x)
            # break

        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        x_list = x.transpose([1,0,2]).chunk(1 + self.ngram, 1)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":x.numpy()})         
        if attn is not None:
            attn_list = attn.transpose([1, 0,2]).chunk(1 + self.ngram, 1)
        else:
            attn_list = None
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #     {"x":attn_list.numpy()})         
        return x_list, {'attn': attn_list}
    def cal_finetune_relative_positions(self, real_positions):
        # print(real_positions)
        n_tokens = paddle.shape(real_positions)[-1]
        batch_size = paddle.shape(real_positions)[0]
        if not hasattr(self, '_finetune_i_bucket_main_stream') or self._finetune_i_bucket_main_stream is None or self._finetune_i_bucket_main_stream.place != real_positions.place:
            fake_positions = paddle.arange(1, self.max_target_positions + 1).reshape((1,-1))
            # fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            # print("finetune_i_bucket_main_stream:",finetune_i_bucket_main_stream)
            self._finetune_i_bucket_main_stream = paddle.to_tensor(finetune_i_bucket_main_stream,place=real_positions.place)
            self._finetune_i_bucket_predicting_stream = paddle.to_tensor(finetune_i_bucket_predicting_stream,place=real_positions.place)
        # print("self._finetune_i_bucket_main_stream:", self._finetune_i_bucket_main_stream)
        finetune_i_bucket_main_stream = paddle.expand(self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens],[batch_size,n_tokens,n_tokens])
        # print("finetune_i_bucket_main_stream:",finetune_i_bucket_main_stream)
        finetune_i_bucket_predicting_stream = paddle.concat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2)

        finetune_i_bucket_predicting_stream=paddle.expand(finetune_i_bucket_predicting_stream,[batch_size,
                                                                                               paddle.shape(finetune_i_bucket_predicting_stream)[1],
                                                                                               paddle.shape(finetune_i_bucket_predicting_stream)[2]])
        # print("finetune_i_bucket_predicting_stream:", finetune_i_bucket_predicting_stream)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream
    def cal_pretrain_relative_positions(self, real_positions):
        # main stream
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        T = paddle.shape(real_positions)[1]
        main_stream_relative_positions=paddle.expand(main_stream_relative_positions,shape=[1,T,T])
        # print(np.shape(main_stream_relative_positions),main_stream_relative_positions)
        # [B,T,1]
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main
        # predicting stream
        # input shift
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        predicting_stream_relative_positions=paddle.fluid.layers.concat(input=[real_positions_shift_predicting_stream,real_positions], axis=-1).unsqueeze(1)
        # print(predicting_stream_relative_positions)
        # [B,T, 2*T]
        # predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, real_positions.size(-1),
        #                                                                                    1)
        predicting_stream_relative_positions=paddle.expand(predicting_stream_relative_positions, shape=[1,paddle.shape(real_positions)[-1], paddle.shape(predicting_stream_relative_positions)[-1]])
        # print(predicting_stream_relative_positions)
        # [B,T, 1]
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        # print(predicting_stream_relative_positions)
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(predicting_stream_relative_positions,
                                                                   bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream
    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            # result = result + paddle.lt(n, paddle.zeros_like(n)).int() * num_buckets
            result = result + paddle.cast(n<paddle.zeros_like(n), "uint8") * num_buckets
            n = paddle.abs(n)
        else:
            n = paddle. maximum(n, paddle.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n<max_exact
        val_if_large = max_exact + paddle.log(paddle.cast(n,"float32") / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = paddle.minimum(val_if_large, paddle.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = paddle.cast(val_if_large,"int32")
        result = result + paddle.where(is_small,  paddle.cast(n,"int32"), val_if_large)
        return result
    def buffered_future_mask(self, tensor):
        dim = paddle.shape(tensor)[0]
        if not hasattr(self, '_future_mask') or \
            self._future_mask is None or self._future_mask.place != tensor.place or \
            paddle.shape(self._future_mask)[0] < dim:
                self._future_mask = paddle.triu(paddle.full([dim,dim],fill_value =float('-inf')), 1)
        return self._future_mask[:dim, :dim]
    def buffered_future_mask_ngram(self, tensor):
        dim = paddle.shape(tensor)[0]
        if not hasattr(self, '_ngram_future_mask') or \
            self._ngram_future_mask is None or \
            self._ngram_future_mask.place != tensor.place:
                self._ngram_future_mask = ngram_attention_bias(self.max_target_positions, self.ngram)                
                self._ngram_future_mask=paddle.to_tensor(self._ngram_future_mask,place=tensor.place, dtype=tensor.dtype)

        ngram_future_mask = paddle.concat([self._ngram_future_mask[:, :dim, :dim],self._ngram_future_mask[:, :dim,self.max_target_positions: self.max_target_positions + dim]], 2)
        return ngram_future_mask
    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight.transpose([1,0]))
        else:
            return F.linear(features, self.embed_out.transpose([1,0]))      


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self,num_embeddings: int,embedding_dim: int,padding_idx: int,):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        np.random.seed(666)
        w0=np.random.rand(num_embeddings,embedding_dim).astype(np.float32)
        self.weight.set_value(w0)

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
            (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions=[]
                for src in input:
                    positions.append(list(range(1,len(src)+1)))
                positions=paddle.to_tensor(positions)
                flag=input!=0
                positions=positions*paddle.cast(flag,"uint8")
            real_positions = positions
        else:
            real_positions = positions
        return super().forward(real_positions), real_positions

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def _forward(self, positions):
        # print("self.weight:", self.weight)
        return super().forward(positions)


if __name__=="__main__":
  from scipy.io import loadmat,savemat
#   test_sample=loadmat("../test_sample.mat") #, {'src_tokens':src_tokens,'src_lengths':src_lengths,'prev_output_tokens':prev_output_tokens}
  test_sample=loadmat("/content/drive/MyDrive/ProphetNet_En/test_sample.mat")
  src_tokens=paddle.to_tensor(test_sample['src_tokens'])
  prev_output_tokens=paddle.to_tensor(test_sample['prev_output_tokens'])
  src_lengths=test_sample['src_lengths'][0]

  model = NgramTransformerProphetModel(
    src_vocab_size=30522,
    trg_vocab_size=30522,
    max_length=512,
    num_encoder_layers=12,
    num_decoder_layers=12,
    # n_head=8,
    emb_dim=1024,
    # d_inner_hid=2048,
    dropout=0.1,
    bos_id=0,
    eos_id=1)
  # predict=model(src_tokens=src_tokens,src_lengths=src_lengths,prev_output_tokens=prev_output_tokens)
  # print(model.state_dict().keys())
  # print("predict:",predict)

  # embed_paddle=paddle.nn.Embedding(num_embeddings=30522,
  #           embedding_dim=1024,
  #           padding_idx=0,
  #           weight_attr=paddle.ParamAttr(
  #               initializer=paddle.nn.initializer.Normal(0., 1024**-0.5)))
  # np.random.seed(666)
  # w0=np.random.rand(30522,1024).astype(np.float32)
  # embed_paddle.weight.set_value(w0)
  # TransformerEncoder_paddle=TransformerEncoder(dropout=0.1,max_source_positions=512,embed_tokens=embed_paddle)
  # encoder_out=TransformerEncoder_paddle(src_tokens=src_tokens,src_lengths=src_lengths)

  # #位置嵌入
  # src_length=len(src_tokens)
  # LearnedPositionalEmbedding_paddle=LearnedPositionalEmbedding(513,1024,0)
  # pos_emb, real_positions=LearnedPositionalEmbedding_paddle(src_tokens)
  # print("pos_emb:",pos_emb)






