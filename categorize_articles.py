import tensorflow as tf
import numpy as np
import keras
import tiktoken
import argparse

from gpt_architecture import MultiHeadAttention, FeedForward, TransformerBlock, GPTModel

gpt = tf.keras.models.load_model('model.keras')

tokenizer = tiktoken.get_encoding('gpt2')

labels = ['World', 'Sports', 'Business', 'Sci/Tech']

def classify(text):
    tokenized_text = tf.convert_to_tensor([tokenizer.encode(text)])

    logits = gpt(tokenized_text)[:, -1, :]

    probas = tf.nn.softmax(logits)

    label_idx = np.argmax(probas, axis = -1)
    print(labels[np.squeeze(label_idx)])

parser = argparse.ArgumentParser()

parser.add_argument('--summary', type=str ,default="Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.")

args = parser.parse_args()

classify(text = args.summary)