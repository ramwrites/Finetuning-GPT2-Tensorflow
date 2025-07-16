import os
import json
import requests
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import tiktoken
import pandas as pd
from sklearn.model_selection import train_test_split

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


# weights download code taken from the gpt2 github

model = '124M'

subdir = os.path.join('models', model)
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:

    r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/" + subdir + "/" + filename, stream=True)

    with open(os.path.join(subdir, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

# GPT2 Model architecture

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
    

demo_gpt = GPTModel()

demo_gpt(tf.zeros((1,1024))) # this is to build the model

## Loading the weights of Openai gpt2 to my gpt2 architecture

with open('models/124M/hparams.json') as f:
    hparams = json.load(f)

params = {'blocks': [{} for _ in range(hparams['n_layer'])]}
for name, _ in tf.train.list_variables('models/124M/model.ckpt'):
    # weights of the layer
    variable_array = tf.squeeze(tf.train.load_variable('models/124M/model.ckpt', name))

    variable_name_parts = name.split('/')[1:] # skipping the 'model/' prefix

    target_dict = params
    if variable_name_parts[0].startswith('h'):
        layer_number = int(variable_name_parts[0][1:])
        target_dict = params['blocks'][layer_number]

    for key in variable_name_parts[1:-1]:
        target_dict = target_dict.setdefault(key, {})

    last_key = variable_name_parts[-1]
    target_dict[last_key] = variable_array


# this function doesn't loads weights for the last layer in the gpt cuz the layer is replaced by another layer
def weights_loader(demo_gpt, params):
    demo_gpt.pos_emb.set_weights([params['wpe']])
    demo_gpt.tok_emb.set_weights([params['wte']])
    demo_gpt.final_norm.set_weights([params['g'], params['b']])
    for b in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
        demo_gpt.trf_blocks.layers[b].att.W_key.set_weights([k_w, k_b])
        demo_gpt.trf_blocks.layers[b].att.W_value.set_weights([v_w, v_b])
        demo_gpt.trf_blocks.layers[b].att.W_query.set_weights([q_w, q_b])
        
        out_proj_weights = params["blocks"][b]["attn"]["c_proj"]["w"]
        out_proj_bias = params["blocks"][b]["attn"]["c_proj"]["b"]
        demo_gpt.trf_blocks.layers[b].att.out_proj.set_weights([out_proj_weights, out_proj_bias])
        
        ff_layer0_weights = params['blocks'][b]['mlp']['c_fc']['w']
        ff_layer0_bias = params['blocks'][b]['mlp']['c_fc']['b']
        ff_layer2_weights = params['blocks'][b]['mlp']['c_proj']['w']
        ff_layer2_bias = params['blocks'][b]['mlp']['c_proj']['b']
        demo_gpt.trf_blocks.layers[b].ff.layers.layers[0].set_weights([ff_layer0_weights, ff_layer0_bias])
        demo_gpt.trf_blocks.layers[b].ff.layers.layers[2].set_weights([ff_layer2_weights, ff_layer2_bias])
        
        norm1_beta = params['blocks'][b]['ln_1']['b']
        norm1_gamma = params['blocks'][b]['ln_1']['g']
        norm2_beta = params['blocks'][b]['ln_2']['b']
        norm2_gamma = params['blocks'][b]['ln_2']['g']
        demo_gpt.trf_blocks.layers[b].norm1.set_weights([norm1_gamma, norm1_beta])
        demo_gpt.trf_blocks.layers[b].norm2.set_weights([norm2_gamma, norm2_beta])

weights_loader(demo_gpt, params)

for layer in demo_gpt.layers:
    print(layer)
    layer.trainable = False
    print(layer.trainable)

demo_gpt.trf_blocks.layers[-1].trainable = True
demo_gpt.out_head.trainable = True

## Data loading and preprocessing

df = pd.read_csv('ag_news_test.csv', names=['label', 'title', 'summary'])
# I used the AG news test dataset to train because of the limited compute resources

# this ensures that the train val and test contains same distribution 
train_val, test = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['label'], random_state=42)

train.to_csv('train_data.csv', index=None)
test.to_csv('test_data.csv', index=None)
val.to_csv('val_data.csv', index=None)

train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

del train_data['title']
del val_data['title']
del test_data['title']

tokenizer = tiktoken.get_encoding('gpt2')

def tokenize_and_pad(tokenizer, txt, max_len, pad_token):
    pad_token = [pad_token]
    tokenized_text = [tokenizer.encode(i) for i in txt]
    tokenized_text = [i[:max_len] for i in tokenized_text]
    tokenized_and_padded_text = [i + pad_token*(max_len - len(i)) for i in tokenized_text]

    return tokenized_and_padded_text


val_data.loc[:,'tokenized'] = tokenize_and_pad(tokenizer, val_data.iloc[:,1], 354, 50256)
train_data.loc[:,'tokenized'] = tokenize_and_pad(tokenizer, train_data.iloc[:,1], 354, 50256)
test_data.loc[:,'tokenized'] = tokenize_and_pad(tokenizer, train_data.iloc[:,1], 354, 50256)

del val_data['summary']
del train_data['summary']
del test_data['summary']

# this is to make the labels in the dataset start from 0
val_data.loc[:, 'label'] = val_data.loc[:, 'label'] - 1
train_data.loc[:, 'label'] = train_data.loc[:, 'label'] - 1
test_data.loc[:, 'label'] = train_data.loc[:, 'label'] - 1

def df_to_dataset(df):
    input_ids = tf.constant(df['tokenized'].tolist(), dtype=tf.int32)
    labels = tf.constant(df['classes'].tolist(), dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((input_ids, labels))

    return dataset

val = df_to_dataset(val_data)
train = df_to_dataset(train_data)
test = df_to_dataset(test_data)

batched_val = val.shuffle(buffer_size = val.cardinality()).batch(4)
batched_train = train.shuffle(buffer_size = train.cardinality()).batch(4)
batched_test = test.shuffle(buffer_size = test.cardinality()).batch(4)

# Loss Functions

def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)[:,-1,:]
    loss = tf.keras.losses.sparse_categorical_crossentropy(target_batch, logits, from_logits=True)
    return tf.reduce_mean(loss)

def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0
    if num_batches == None:
        num_batches = data_loader.cardinality()
    for i, (inp_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(inp_batch, target_batch, model)
            total_loss += loss
    return total_loss/tf.cast(num_batches, tf.float32)

def class_accuracy_loader(dataset, model, num_batches=None):
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    if num_batches == None:
        num_batches = dataset.cardinality()
    for i, (inp_batch, target_batch) in enumerate(dataset):
        if i<num_batches:
            logits = model(inp_batch, training=False)[:,-1,:]

        accuracy_metric.update_state(target_batch, logits)

    accuracy = accuracy_metric.result().numpy()
    return accuracy

def evaluate(train, val, model, eval_iter=None):
    train_loss = calc_loss_loader(train, model, num_batches=eval_iter)
    val_loss = calc_loss_loader(val, model, num_batches=eval_iter)

    return train_loss, val_loss

def sample_batch_dataset(dataset, batches=50):
  dataset = dataset.shuffle(buffer_size=dataset.cardinality())
  sampled_dataset = dataset.take(batches)

  return sampled_dataset

## training

opt = tf.keras.optimizers.AdamW(learning_rate=1e-5)
# after training for few epoches i realized that 1e-5 is very low so i changed it to 5e-5 and saw faster convergence

train_losses, val_losses = [], []

def train_func(model, train_dataset, val_dataset, eval_freq, eval_iter=None):
    steps = 0
    for (i, j) in train_dataset:
        with tf.GradientTape() as tape:
            loss = calc_loss_batch(i, j, model)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        steps += 1
        if steps%10 ==0:
          print(steps)
        if steps % eval_freq == 0:
            out_layer_weights = model.layers[-1].get_weights()
            trf_layer_weights = model.trf_blocks.layers[-1].get_weights()
            np.save(f'saved/weights_out{steps}.npy', out_layer_weights)
            np.save(f'saved/weights_trf{steps}.npy', trf_layer_weights)
            train_sample = sample_batch_dataset(train_dataset)
            val_sample = sample_batch_dataset(val_dataset)
            train_loss, val_loss = evaluate(train_sample, val_sample, model, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(steps)
            print(f'train_loss: {train_loss}')
            print(f'val_loss: {val_loss}')

num_epochs = 10
# after few epoches of training I added some new data to the training dataset for improved generalization
for i in range(num_epochs):
    print(f'epoch: {i}')
    train_func(demo_gpt, batched_train, batched_val, 200)