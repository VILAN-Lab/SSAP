#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import csv
import json
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import *
from pplm_classification_head import ClassificationHead

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 768


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size=None,
            pretrained_model="../ckpt/final_title",
            classifier_head=None,
            cached_mode=False,
            device='cpu'
    ):
        super(Discriminator, self).__init__()
        # if pretrained_model.startswith("gpt2"):
        #     self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        #     self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        #     self.embed_size = self.encoder.transformer.config.hidden_size
        # elif pretrained_model.startswith("bert"):
        #     self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        #     self.encoder = BertModel.from_pretrained(pretrained_model)
        #     self.embed_size = self.encoder.config.hidden_size
        # else:
        #     raise ValueError(
        #         "{} model not yet supported".format(pretrained_model)
        #     )
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size

        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(x)
        else:
            # for bert
            hidden, _ = self.encoder(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob



def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))

def load_clasifier_head(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params


def load_discriminator(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    # classifier_head, meta_param = load_classifier_head(
    #     weights_path, meta_path, device
    # )
    discriminator =  Discriminator(
        pretrained_model=meta_params['pretrained_model'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_params


#  "sentiment": {
#         "path": "./ckpt/SST_classifier_head_epoch_85.pt",
#         #"url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
#         "class_size": 5,
#         "embed_size": 768,
#         "class_vocab": {"positive": 0, "negative": 1, "very positive": 2, "very negative": 3, "neutral": 4},
#         "default_class": 4,
#         "pretrained_model": "../ckpt/final_title",
#     },

if __name__ == "__main__":
    idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]

    discriminator, _ = load_discriminator("/home/ubuntu/birdring/Stable_SEG/PPLM/ckpt/SST_classifier_head_epoch_85.pt", "/home/ubuntu/birdring/Stable_SEG/PPLM/ckpt/SST_classifier_head_meta.json", device='cpu')
    # discriminator = Discriminator(
    #             class_size=len(idx2class),
    #             #pretrained_model=pretrained_model,
    #             cached_mode=cached,
    #             device=device
    #         ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("../ckpt/final_title")
    file_data = ""
    with open("/home/ubuntu/birdring/SSAP/data/roc_test.response", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # s = "guitar lessons .	john was learning to play the guitar .	his roommate saw his guitar in the corner .	john's roommate was a pretty good guitar player .	he offered to help john play the guitar ."
            ss = line.split('\t')
            _ = np.array([0.,0.,0.,0.,0.])
            sum_log_probs = torch.from_numpy(_)

            for _, i in enumerate(ss) :
                #print(_, i)
                #if(_ != 5) : continue
                seq = tokenizer.encode(i)
                seq = torch.tensor([seq], device='cpu', dtype=torch.long)
                log_probs = discriminator(seq)[0]
                max_prob = 0
                max_c = ""
                for c, log_prob in zip(idx2class, log_probs) :
                    if(math.exp(log_prob) > max_prob) :
                        max_prob = math.exp(log_prob)
                        max_c = c
                print(_, max_c)
                file_data = file_data + max_c + '\t'
            file_data += '\n'
    with open("/home/ubuntu/birdring/SSAP/result/sentiment_eval/sentiment_predict_test.response","w",encoding="utf-8") as f:
#with open("../sentiment_eval/endding_pplm.txt","w",encoding="utf-8") as f:
        f.write(file_data)