import os
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import numpy as np


# ref_file = '../data/roc_test.response'
# hyp_file = '/home/ubuntu/birdring/SSAP/result/final/pplm_0524.txt'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)
# lm_tokenizer = tokenizer_class.from_pretrained('gpt2')
# lm_model = model_class.from_pretrained('gpt2')
# lm_model.to(device)
# lm_model.eval()

# lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
# path = '../ckpt/roc_yaki/pytorch_model.bin'
# lm_model_state_dict = torch.load(path)
# lm_model.load_state_dict(lm_model_state_dict)
# lm_model.to(device)
# lm_model.eval()
lm_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

def calculate_ppl_gpt(sentence_batch):
    tokenized_ids = [None for i in range(len(sentence_batch))]

    for i in range(len(sentence_batch)):
        tokenized_ids[i] = lm_tokenizer.encode(sentence_batch[i])
        
    sen_lengths = [len(x) for x in tokenized_ids]
    max_sen_lenght = max(sen_lengths)
    
    n_batch = len(sentence_batch)
    input_ids = np.zeros(shape=(n_batch, max_sen_lenght), dtype=np.int64)
    lm_labels = np.full(shape=(n_batch, max_sen_lenght), fill_value=-1)
    
    for i, tokens in enumerate(tokenized_ids):
        input_ids[i, :len(tokens)] = tokens
        lm_labels[i, :len(tokens)-1] = tokens[1:] 
    
    input_ids = torch.tensor(input_ids).to(device)
    lm_labels = torch.tensor(lm_labels).to(device)
    with torch.no_grad():
        lm_pred = lm_model(input_ids)
    loss_val = lm_loss(lm_pred[0].view(-1, lm_pred[0].size(-1)), lm_labels.view(-1))
    normalized_loss = loss_val.view(n_batch,-1).sum(dim= -1) / torch.tensor(sen_lengths, dtype=torch.float32).to(device)
    #normalized_loss = loss_val.view(n_batch,-1).sum(dim= -1)
    ppl = torch.exp(normalized_loss)
    return  ppl.tolist() 

def get_ppl():
    hyp_data_open = open(hyp_file, "r")
    hyp_data_dataset = hyp_data_open.readlines()
    hyp_len = len(hyp_data_dataset)
    hyp_data_open.close()

    hyp_PPL = 0
    for k in range(hyp_len):
        out_sen = hyp_data_dataset[k].strip()
        if len(out_sen) != 0:
            sample_PPL = calculate_ppl_gpt([out_sen])
            hyp_PPL += sample_PPL[0]


    total_PPL = (hyp_PPL) / (hyp_len)
    print('PPL: {}'.format(total_PPL))


# NLG-eval
dist = os.popen("python /home/ubuntu/birdring/SSAP/result/Distinct-N/bin/distinct_metric.py --hypothesis=" + hyp_file)
result = os.popen("nlg-eval --references=" + ref_file + " --hypothesis=" + hyp_file)
print(dist.read())
print(result.read())

# PPL
# init()
#get_ppl()
