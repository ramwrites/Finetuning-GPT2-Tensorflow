# GPT2 Model architecture
import tensorflow as tf
import numpy as np
import keras

cfg = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True,       # Query-Key-Value bias
    "num_classes" : 4       # Number of categories in dataset
}

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False, name=None):
        super().__init__(name=name)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = int(d_out//num_heads)
        self.W_key = tf.keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.W_query = tf.keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.W_value = tf.keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.tril = tf.linalg.band_part(tf.Variable(tf.ones((context_length, context_length)), 
                                                    trainable=False), 
                                        num_lower=-1, 
                                        num_upper=0)
        self.out_proj = tf.keras.layers.Dense(d_out)

    def call(self, x, training = True):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = tf.reshape(keys, (b, num_tokens, self.num_heads, self.head_dim))
        queries = tf.reshape(queries, (b, num_tokens, self.num_heads, self.head_dim))
        values = tf.reshape(values, (b, num_tokens, self.num_heads, self.head_dim))
        
        keys = tf.transpose(keys, perm=[0,2,1,3])
        queries = tf.transpose(queries, perm=[0,2,1,3])
        values = tf.transpose(values, perm=[0,2,1,3])

        atten_scores = queries @ tf.transpose(keys, perm=[0, 1, 3, 2]) * keys.shape[-1]**-0.5
        mask_bool = self.tril[:num_tokens, :num_tokens]

        atten_scores = tf.where(mask_bool == 0, tf.fill(atten_scores.shape, float('-inf')), atten_scores)

        atten_weights = tf.nn.softmax(atten_scores, axis = -1)
        atten_weights = self.dropout(atten_weights, training = training)

        context_vec = tf.transpose(atten_weights @ values, perm = [0,2,1,3])

        context_vec = tf.reshape(context_vec, (b, num_tokens, self.d_out))
        context_vec = self.out_proj(context_vec)
        return context_vec
    

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg['emb_dim']*4),
                tf.keras.layers.Lambda(tf.keras.activations.gelu),
                tf.keras.layers.Dense(cfg['emb_dim']),
            ]
        )
    def call(self, x):
        return self.layers(x)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qkv_bias'])
        self.ff = FeedForward(cfg)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.drop_resid = tf.keras.layers.Dropout(cfg['drop_rate'])

    def call(self, x, training = True):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, training = training)
        x = self.drop_resid(x, training = training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x, training = training)
        x = x + shortcut
        return x
    
@keras.saving.register_keras_serializable()
class GPTModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global cfg
        self.tok_emb = tf.keras.layers.Embedding(input_dim=cfg['vocab_size'], output_dim=cfg['emb_dim'])
        self.pos_emb = tf.keras.layers.Embedding(input_dim=cfg['context_length'], output_dim=cfg['emb_dim'])
        self.drop_emb = tf.keras.layers.Dropout(cfg['drop_rate'])
        self.trf_blocks = tf.keras.Sequential(
            [TransformerBlock(cfg) for _ in range(cfg['n_layers'])] 
        )
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.out_head = tf.keras.layers.Dense(cfg['num_classes'], use_bias=False)
    def call(self, in_idx, training = True):
        batch, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(tf.range(seq_len))
        x = tok_emb + pos_emb
        x = self.drop_emb(x, training=training)
        x = self.trf_blocks(x, training=training)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits