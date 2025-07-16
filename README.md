# Classification Finetuning GPT-2

This repository contains the code I've written for classification finetuning GPT2.

I rebuilt the GPT-2 architecture from scratch using Tensorflow and downloaded the pre-trained weights from OpenAI's GPT-2 124M model using the code mentioned in the official GPT2 [Github](https://github.com/openai/gpt-2) repo and loaded them. I replaced the output head with a custom classification layer (output dimension = 4).

In this project, I fine-tuned GPT-2 on a subset of the AG News dataset to classify news articles into one of four categories:

* World

* Sports

* Business

* Sci/Tech

I freezed weights of all layers except the last Transformer block and the output head before finetuning the model.

After fine-tuning on the AG News dataset, the model achieved a test accuracy of 91%.

To train the model, run the following script:

```bash
python .\train.py
```

After training, the model can be used classifying news articles. The script `categorize_articles.py` takes a news article summary and predicts its category.

For example, to classify a sample news summary, run:

```bash
python .\categorize_articles.py --summary "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
```
```bash
Output:

Business
```
