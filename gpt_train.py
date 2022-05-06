import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from tqdm import tqdm
import torch.optim as optim
import os
import random

from transformers import *
model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)

tokenizer = tokenizer_class.from_pretrained('gpt2')
tokenizer.add_tokens("<kw>")
tokenizer.add_tokens("</kw>")
model = model_class.from_pretrained('gpt2').cuda()
model.train()   

def main():
    data_path = "./data/"
    data_train_path = data_path + "roc_train.txt"

    roc_open = open(data_train_path, "r")
    roc_dataset = roc_open.readlines()
    roc_len = len(roc_dataset)
    roc_open.close()

    epoch = 5
    stop_point = roc_len * epoch    
    
    # Parameters:
    lr = 1e-3
    max_grad_norm = 1.0
    num_total_steps = stop_point # 1000
    num_warmup_steps = int(stop_point/10) # 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

    #lm_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lm_loss = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps =num_total_steps)  # PyTorch scheduler    

    #torch.cuda.empty_cache()
    for start in tqdm(range(stop_point)):
        """data start point"""
        roc_start = start % roc_len

        """data setting"""
        roc_sentence = roc_dataset[roc_start].strip()

        """data input"""
        sentence = roc_sentence.lower()
        #sentence += ' <|endoftext|>'
        
        sen_idx = torch.tensor(tokenizer.encode(sentence)).cuda()
       
        output = model(sen_idx)
        
        if len(sen_idx) == 1: continue
        target = sen_idx[1:]
        pred = output[0][:-1,:]  

        # target = sen_idx
        # pred = output[0]         
        #print(pred, target)            

        loss = lm_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(loss)
        if (start+1) % roc_len == 0:
            random.shuffle(roc_dataset)
            print("train epoch done: ")
            print((start+1) % roc_len)
            #save_model((start + 1) // roc_len)       

    save_model('./ckpt/roc') # final_model
    
    
def save_model(name):
    if not os.path.exists(str(name)+'/'): os.makedirs(str(name)+'/')
    model.save_pretrained('./'+str(name))
    tokenizer.save_pretrained('./'+str(name))    

if __name__ == '__main__':
    main()
        