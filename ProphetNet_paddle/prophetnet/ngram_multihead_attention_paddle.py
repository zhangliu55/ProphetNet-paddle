import paddle
import paddlenlp
from paddlenlp.transformers import ErnieModel, ErnieTokenizer, BertTokenizer
from paddlenlp.transformers import *
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import math
from scipy.io import loadmat,savemat

# import torch


def ngram_attention_bias(length, num_skip):
        bias_result = []
        for n_skip in range(num_skip):
            bias_n_skip = []
            for i in range(length):
                bias_this = [float('-inf')] * (2 * length)
                bias_this[length+i] = 0
                first_k = i - n_skip
                first_k = first_k if first_k > 0 else 0
                for j in range(first_k+1):
                    bias_this[j] = 0
                bias_n_skip.append(bias_this)
            bias_result.append(bias_n_skip)
        # print('ngram_attention_bias:',paddle.to_tensor(np.array(bias_result, dtype=np.float32)))
        return paddle.to_tensor(np.array(bias_result, dtype=np.float32))

class NgramMultiheadAttention(nn.Layer):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False,ngram=2, num_buckets=32, relative_max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.ngram = ngram

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'
        init_parattr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_init_parattr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        self.relative_linear = nn.Linear(embed_dim, num_buckets * num_heads,
         weight_attr=init_parattr,bias_attr=bias_init_parattr)
    
        if self.qkv_same_dim:
            # self.in_proj_weight = paddle.ParamAttr(torch.Tensor(3 * embed_dim, embed_dim))
            self.in_proj_weight =paddle.create_parameter(shape=(3 * embed_dim, embed_dim),dtype="float32",default_initializer=nn.initializer.Constant(1))

        else:
            # self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.k_proj_weight =paddle.create_parameter(shape=(embed_dim, self.kdim),dtype="float32",default_initializer=nn.initializer.Constant(1))
            # self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.v_proj_weight = paddle.create_parameter(shape=(embed_dim, self.kdim),dtype="float32",default_initializer=nn.initializer.Constant(1))
            # self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.q_proj_weight = paddle.create_parameter(shape=(embed_dim, embed_dim),dtype="float32",default_initializer=nn.initializer.Constant(1))
        if bias:
            # self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.in_proj_bias = paddle.create_parameter(shape=(3 * embed_dim,),dtype="float32",is_bias=True ,default_initializer=nn.initializer.Constant())
        else:
            self.register_parameter('in_proj_bias', None)
        # self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=bias)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,
         weight_attr=init_parattr,bias_attr=bias_init_parattr)

        if add_bias_kv:
            # self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            # self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_k = paddle.create_parameter(shape=(1, 1, embed_dim),dtype="float32",is_bias=True, default_initializer=nn.initializer.Constant())
            self.bias_v = paddle.create_parameter(shape=(1, 1, embed_dim),dtype="float32",is_bias=True, default_initializer=nn.initializer.Constant())
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self.reset_parameters()

        self.onnx_trace = False

    # def prepare_for_onnx_export_(self):
    #     self.onnx_trace = True
    #
    # def reset_parameters(self):
    #     if self.qkv_same_dim:
    #         nn.init.xavier_uniform_(self.in_proj_weight)
    #     else:
    #         nn.init.xavier_uniform_(self.k_proj_weight)
    #         nn.init.xavier_uniform_(self.v_proj_weight)
    #         nn.init.xavier_uniform_(self.q_proj_weight)
    #
    #     nn.init.xavier_uniform_(self.out_proj.weight)
    #     if self.in_proj_bias is not None:
    #         nn.init.constant_(self.in_proj_bias, 0.)
    #         nn.init.constant_(self.out_proj.bias, 0.)
    #     if self.bias_k is not None:
    #         nn.init.xavier_normal_(self.bias_k)
    #     if self.bias_v is not None:
    #         nn.init.xavier_normal_(self.bias_v)

    # def _relative_positions_bucket(self, relative_positions, bidirectional=False):
    #     num_buckets = self.num_buckets
    #     max_distance = self.relative_max_distance
    #     n = -relative_positions
    #     result = 0
    #     if bidirectional:
    #         num_buckets = num_buckets // 2
    #         result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
    #         n = torch.abs(n)
    #     else:
    #         n = torch.max(n, torch.zeros_like(n))
    #     max_exact = num_buckets // 2
    #     is_small = torch.lt(n, max_exact)
    #     val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
    #                 num_buckets - max_exact)
    #     val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
    #     val_if_large = val_if_large.int()
    #     result = result + torch.where(is_small, n.int(), val_if_large)
    #     return result


    def main_stream_relative_logits(self,query, attn_weights, real_positions,i_bucket_main_stream):
        # input query [T,B,C]
        # input attn_weights [T*head,T,S]
        # input real_positions [B,T] or [1,1]
    
        T,B,_ = query.shape
        # print(T,B,_)
        S = attn_weights.shape[-1]
        # print(S)
        if i_bucket_main_stream is not None:
            i_buckets = i_bucket_main_stream
        else:
            # [B,T,S]
            # relative_positions = torch.arange(1, S+1).unsqueeze(0).unsqueeze(0).repeat(B,T,1).to(real_positions.device)
            relative_positions = paddle.arange(1, S+1).unsqueeze(0).unsqueeze(0)
            relative_positions = paddle.to_tensor(relative_positions.broadcast_to([B,T,relative_positions.shape[2]]),place=real_positions.place)
            print(np.shape(relative_positions))
            # [B,T,1]
            # real_positions = real_positions.unsqueeze(0).repeat(B,T,1)
            real_positions = real_positions.unsqueeze(0)
            print(np.shape(real_positions))
            real_positions = real_positions.broadcast_to([B,T,real_positions.shape[2]])
            # [B,T,S]
            print(np.shape(relative_positions),np.shape(real_positions))
            relative_positions = relative_positions - real_positions
            # [B,T,T]
            i_buckets = self._relative_positions_bucket(relative_positions, False)
    
        # # [B,T,C]
        # query = query.transpose(0,1)
        query = query.transpose([1,0,2])
        # # [B,T,Buckets*head]
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":query.numpy()})  
        values = self.relative_linear(query)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":values.numpy()})         
        # # [B,T,Buckets,head]
        # values = values.view(values.size(0),values.size(1),self.num_buckets, self.num_heads)
        values = values.reshape([values.shape[0],values.shape[1],self.num_buckets, self.num_heads])
        
        # # [B,head,Buckets,T]
        # values = values.transpose(1,3)
        values = values.transpose([0,3,2,1])
        # # [B,head,T,Buckets]
        # values = values.transpose(2,3)
        values = values.transpose([0,1,3,2])
        # # [B*head,T,Buckets]
        # values = values.reshape(attn_weights.size(0),attn_weights.size(1),-1)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":values.numpy()})         
        values = values.reshape([attn_weights.shape[0],attn_weights.shape[1],-1])
    
        # => [B,head*T,T] => [B*head,T,T]
        # i_buckets = i_buckets.repeat(1,self.num_heads,1).view(attn_weights.size(0),attn_weights.size(1),-1)
        i_buckets=paddle.concat([i_buckets for i in range(self.num_heads)],axis=1)
        i_buckets=i_buckets.reshape([attn_weights.shape[0],attn_weights.shape[1],-1])

        # [B*head*T,Buckets]
        values = values.reshape([-1, values.shape[-1]])
        # [B*head*T,T]
        # i_buckets = i_buckets.reshape([-1, i_buckets.shape[-1]]).long()
        i_buckets = i_buckets.reshape([-1, i_buckets.shape[-1]]).astype('int64')
        # [B*head*T,T]
        # print(values,i_buckets)
        # result = paddle.gather(x=values,index=i_buckets, axis=1)
        result=paddle.concat([paddle.gather(x=values[i],index=index, axis=0).unsqueeze(0) for i,index in enumerate(i_buckets)],axis=0)
        # print(result)
        # [B*head,T,T]
        result = result.reshape([attn_weights.shape[0],attn_weights.shape[1],-1])
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":result.numpy()})     
        return result

    def ngram_relative_logits(self, query, attn_weights, real_positions, i_bucket_relative_stream):
        # input query [ngram, T,B,C]
        # input attn_weights [ngram, B*head,T,S]
        # input real_positions [B,T] or [1,1]
        # input i_bucket_relative_stream [B,T, 2*T] or None
    
        N, T, B, _ = query.shape
        _, BH, _, S = attn_weights.shape
    
        if i_bucket_relative_stream is not None:
            i_buckets = i_bucket_relative_stream
        else:
            # [B,T,S]
            assert real_positions[0][0] == S - 1, 'memory position is 1 2 3 4 5(S-1)'
            relative_positions = torch.arange(0, S).unsqueeze(0).unsqueeze(0).repeat(B,T,1).to(real_positions.device)
            # print('relative_positions', relative_positions)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B,T,1)
            relative_positions = relative_positions
            # [B,T,2*T] or [B,T,S]
            relative_positions = relative_positions - real_positions
            i_buckets = self._relative_positions_bucket(relative_positions, False)
    
        # [ngram, B, T, C]
        query = query.transpose([0,2,1,3])
        # [ngram, B, T, bucket*head]
        values = self.relative_linear(query)
        
        # [ngram, B, T, bucket, head]
        values = values.reshape([*values.shape[:-1],self.num_buckets, self.num_heads])
        # [ngram, B, head, T, bucket]
        values = values.transpose([0, 1, 4, 2, 3])
        # [ngram*B*head, T, bucket]
        values = values.reshape([N*BH,T,-1])
    
        # [ngram, B, head*T, S]
        # i_buckets = i_buckets.unsqueeze(0).repeat(N,1,self.num_heads,1)
        i_buckets = i_buckets.unsqueeze(0)
        i_buckets = i_buckets.broadcast_to([N,i_buckets.shape[1],i_buckets.shape[2],i_buckets.shape[3]])
        i_buckets = paddle.concat([i_buckets for i in range(self.num_heads)],axis=2)
     
        values = values.reshape([-1, values.shape[-1]])
        i_buckets = i_buckets.reshape([-1, i_buckets.shape[-1]]).astype("int64")
        # [ngram*B*head*T, S]
        # result = torch.gather(values,dim=1,index=i_buckets)
        result=paddle.concat([paddle.gather(x=values[i],index=index, axis=0).unsqueeze(0) for i,index in enumerate(i_buckets)],axis=0)

        # [ngram, B*head, T, S]
        result = result.reshape([N, BH , T, -1])
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":values.numpy(),
        #    "y":result.numpy()})    
        return result

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False,
                self_attn_mask=None,
                ngram_mask_matrix=None,
                i_buckets_main_stream=None,
                i_bucket_relative_stream=None,
                real_positions=None
                ):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # print("query:", query)
        # print("key:", key)
        # print("value:", value)
        tgt_len, bsz, embed_dim = paddle.shape(query)
        assert embed_dim == self.embed_dim
        assert list(paddle.shape(query)) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        # print("query:",query)
        q, k, v = self.in_proj_qkv(query)
        # print("q:",q.shape,q)
        q *= self.scaling
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":v.numpy()})
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = paddle.concat([k, self.bias_k.broadcast_to([bias_k.shape[0], bsz, bias_k.shape[2]])])
            v = paddle.concat([v, self.bias_v.broadcast_to([bias_k.shape[0], bsz, bias_k.shape[2]])])
        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # print("q:",q.shape,q)
        q = q.reshape([tgt_len, bsz * self.num_heads, self.head_dim]).transpose([1,0,2])
        # print("q:",q.shape,q)

        if k is not None:
            # k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1,0,2])
        if v is not None:
            # v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1,0,2])
        
        # h: 输入的隐状态
        h_list = query.chunk(1 + self.ngram,  axis=0)
        
        q_list = q.chunk(1 + self.ngram,  axis=1)
        k_list = k.chunk(1 + self.ngram,  axis=1)
        v_list = v.chunk(1 + self.ngram,  axis=1)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":v_list[2]})        
        h_main, h_predict_list = h_list[0], h_list[1:]
        q_main, q_predict_list = q_list[0], q_list[1:]
        k_main, k_predict_list = k_list[0], k_list[1:]
        v_main, v_predict_list = v_list[0], v_list[1:]
        # print(np.shape(h_main),np.shape(q_main),np.shape(k_main),np.shape(v_main))
        # print(np.shape(h_predict_list),np.shape(q_predict_list),np.shape(k_predict_list),np.shape(v_predict_list))
        # print(h_predict_list,q_predict_list)
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].reshape(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    assert False, 'static_kv not supprt in ngram decoder'
                    k = prev_key
                else:
                    k_main = paddle.concat((prev_key, k_main),  axis=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].reshape(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v_main = paddle.concat((prev_value, v_main), axis=1)
            saved_state['prev_key'] = k_main.reshape(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v_main.reshape(bsz, self.num_heads, -1, self.head_dim)
        
            self._set_input_buffer(incremental_state, saved_state)
        
        real_tgt_len = tgt_len // (1 + self.ngram)
        
        # attn_weights_main = torch.bmm(q_main, k_main.transpose_to(1, 2))
        attn_weights_main = paddle.bmm(q_main, k_main.transpose([0,2,1]))
        # print("attn_weights_main:",np.shape(attn_weights_main),attn_weights_main)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":attn_weights_main.numpy()})         
        main_relative_logits = self.main_stream_relative_logits(h_main, attn_weights_main,real_positions, i_buckets_main_stream)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":main_relative_logits.numpy()})         
        # print("main_relative_logits:",np.shape(main_relative_logits),main_relative_logits)
        attn_weights_main = attn_weights_main + main_relative_logits
        
        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.unsqueeze(0)
            attn_weights_main = attn_weights_main + self_attn_mask
        # print(attn_weights_main)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":attn_weights_main.numpy()})        
        # attn_weights_main = utils.softmax(
        #     attn_weights_main, dim=-1, onnx_trace=self.onnx_trace,
        # ).type_as(attn_weights_main)
        attn_weights_main=nn.functional.softmax( x=attn_weights_main, axis=-1,dtype=attn_weights_main.dtype)
        attn_weights_main = F.dropout(attn_weights_main, p=self.dropout, training=self.training)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":attn_weights_main.numpy()})         
        attn_main = paddle.bmm(attn_weights_main, v_main)       
        # print(np.shape(attn_main))
        # attn_main = attn_main.transpose(0, 1).contiguous().view(1, real_tgt_len, bsz, embed_dim)
        attn_main = attn_main.transpose([1, 0,2]).reshape([1, real_tgt_len, bsz, embed_dim])       
        attn_main = self.out_proj(attn_main)
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":attn_main.numpy()})         
        
        # [ngram, B*head, T, c]
        # q_ngram = torch.cat(q_predict_list, 0).view(self.ngram, -1, real_tgt_len, self.head_dim)
        q_ngram = paddle.concat(q_predict_list, axis=0).reshape([self.ngram, -1, real_tgt_len, self.head_dim])
        # [ngram, B*head, 2*T, c]
        k_ngram = paddle.concat([paddle.concat([k_main, k_p], axis=1).unsqueeze(0) for k_p in k_predict_list], 0)
        # below code slower than above for loop
        # k_ngram = torch.cat([k_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(k_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)
        
        # [ngram, T, B, C]
        h_ngram = paddle.concat(h_predict_list, axis=0).reshape([self.ngram, real_tgt_len, bsz, embed_dim])
        
        # [ngram, B*head, 2*T, c]
        v_ngram =  paddle.concat([ paddle.concat([v_main, v_p], axis=1).unsqueeze(0) for v_p in v_predict_list], 0)
        # below code slower than above for loop
        # v_ngram = torch.cat([v_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(v_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)
        #
        # [ngram, B*head, T, 2*T]
        # attn_weights_ngram = torch.einsum('nbtc,nbsc->nbts', (q_ngram, k_ngram))
        attn_weights_ngram=paddlenlp.ops.einsum('nbtc,nbsc->nbts', (q_ngram, k_ngram))
        # gram=q_ngram.shape[0]
        # q_ngram=q_ngram.reshape([-1,q_ngram.shape[2],q_ngram.shape[3]])
        # k_ngram=k_ngram.reshape([-1,k_ngram.shape[2],k_ngram.shape[3]]).transpose([0,2,1])
        # attn_weights_ngram=paddle.bmm(q_ngram,k_ngram).reshape([gram,-1,q_ngram.shape[1],k_ngram.shape[2]])
        # print(np.shape(attn_weights_ngram))
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":i_bucket_relative_stream.numpy()}) 
        # [ngram, B*head, T, S]
        predict_relative_logits = self.ngram_relative_logits(h_ngram, attn_weights_ngram, real_positions, i_bucket_relative_stream)
        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = attn_weights_ngram + predict_relative_logits
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":predict_relative_logits.numpy(),
        #    "y":attn_weights_ngram.numpy()})         
        if ngram_mask_matrix is not None:
            ngram_mask_matrix = ngram_mask_matrix.unsqueeze(1)
            attn_weights_ngram = attn_weights_ngram + ngram_mask_matrix
        
        # attn_weights_ngram = utils.softmax(
        #     attn_weights_ngram, dim=-1, onnx_trace=self.onnx_trace,
        # ).type_as(attn_weights_ngram)
        attn_weights_ngram=nn.functional.softmax( x=attn_weights_ngram, axis=-1,dtype=attn_weights_ngram.dtype)
        attn_weights_ngram = F.dropout(attn_weights_ngram, p=self.dropout, training=self.training)
        
        # [ngram, B*head, T, c]
        # attn_ngram = torch.einsum('nbts,nbsc->nbtc', (attn_weights_ngram, v_ngram))
        # gram=attn_weights_ngram.shape[0]
        # attn_weights_ngram=attn_weights_ngram.reshape([-1,attn_weights_ngram.shape[2],attn_weights_ngram.shape[3]])
        # v_ngram=v_ngram.reshape([-1,v_ngram.shape[2],v_ngram.shape[3]])
        # attn_ngram=paddle.bmm(attn_weights_ngram,v_ngram).reshape([gram,-1,attn_weights_ngram.shape[1],v_ngram.shape[2]])
        attn_ngram=paddlenlp.ops.einsum('nbts,nbsc->nbtc', (attn_weights_ngram, v_ngram))
        # print(attn_ngram.shape)


        # [ngram, T, B, C]
        # attn_ngram = attn_ngram.transpose(1, 2).contiguous().view(self.ngram, real_tgt_len, bsz, embed_dim)
        attn_ngram = attn_ngram.transpose([0,2,1,3]).reshape([self.ngram, real_tgt_len, bsz, embed_dim])
        attn_ngram = self.out_proj(attn_ngram)
        
        attn_result = []
        attn_result.append(attn_main)
        attn_result.append(attn_ngram)
        
        # [1+ngram*T, B, C]
        attn = paddle.concat(attn_result, axis=0).reshape([-1, bsz, embed_dim])
        # savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        #   {"x":attn_ngram.numpy(),
        #    "y":attn.numpy()})         
        return attn, None

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3,  axis=-1)
    #
    # def in_proj_q(self, query):
    #     if self.qkv_same_dim:
    #         return self._in_proj(query, end=self.embed_dim)
    #     else:
    #         bias = self.in_proj_bias
    #         if bias is not None:
    #             bias = bias[:self.embed_dim]
    #         return F.linear(query, self.q_proj_weight, bias)
    #
    # def in_proj_k(self, key):
    #     if self.qkv_same_dim:
    #         return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
    #     else:
    #         weight = self.k_proj_weight
    #         bias = self.in_proj_bias
    #         if bias is not None:
    #             bias = bias[self.embed_dim:2 * self.embed_dim]
    #         return F.linear(key, weight, bias)
    #
    # def in_proj_v(self, value):
    #     if self.qkv_same_dim:
    #         return self._in_proj(value, start=2 * self.embed_dim)
    #     else:
    #         weight = self.v_proj_weight
    #         bias = self.in_proj_bias
    #         if bias is not None:
    #             bias = bias[2 * self.embed_dim:]
    #         return F.linear(value, weight, bias)
    #
    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight      
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]

        # np.random.seed(666)
        # input=np.random.rand(207, 2, 1024).astype(np.float32)
        # input=paddle.to_tensor(input)

        return F.linear(input, paddle.transpose(weight,[1,0]), bias)
    #
    # def reorder_incremental_state(self, incremental_state, new_order):
    #     """Reorder buffered internal state (for incremental generation)."""
    #     input_buffer = self._get_input_buffer(incremental_state)
    #     if input_buffer is not None:
    #         for k in input_buffer.keys():
    #             input_buffer[k] = input_buffer[k].index_select(0, new_order)
    #         self._set_input_buffer(incremental_state, input_buffer)
    #
    # def _get_input_buffer(self, incremental_state):
    #     return utils.get_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #     ) or {}
    #
    # def _set_input_buffer(self, incremental_state, buffer):
    #     utils.set_incremental_state(
    #         self,
    #         incremental_state,
    #         'attn_state',
    #         buffer,
    #     )



if __name__=="__main__":
    from scipy.io import loadmat,savemat
    # from prophetnet.ngram_multihead_attention import NgramMultiheadAttention, ngram_attention_bias
    test_sample_ngram_self_attn=loadmat("/content/drive/MyDrive/ProphetNet_En/test_sample_ngram_self_attn.mat")

    ngram_self_attn=NgramMultiheadAttention(
                1024,
                16,
                dropout=0.1,
                add_bias_kv=False,
                add_zero_attn=False,
                self_attention=True,
                ngram=2
            )
    x, attn = ngram_self_attn(
        query=paddle.to_tensor(test_sample_ngram_self_attn["x"].astype(np.float32)),
        key=paddle.to_tensor(test_sample_ngram_self_attn["x"].astype(np.float32)),
        value=paddle.to_tensor(test_sample_ngram_self_attn["x"].astype(np.float32)),
        incremental_state=None,
        need_weights=False,
        self_attn_mask=paddle.to_tensor(test_sample_ngram_self_attn["self_attn_mask"].astype(np.float32)),
        ngram_mask_matrix=paddle.to_tensor(test_sample_ngram_self_attn["ngram_mask_matrix"].astype(np.float32)),
        i_buckets_main_stream=paddle.to_tensor(test_sample_ngram_self_attn["i_buckets_main_stream"].astype(np.float32)),
        i_bucket_relative_stream=paddle.to_tensor(test_sample_ngram_self_attn["i_bucket_relative_stream"].astype(np.float32)),
        real_positions=paddle.to_tensor(test_sample_ngram_self_attn["real_positions"].astype(np.float32))
    )
    print("x:",x)

    savemat("/content/drive/MyDrive/ProphetNet_En/testData_paddle.mat",
        {"x":x.numpy()}) 


