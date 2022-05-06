import os


file_ending = '/home/ubuntu/birdring/SSAP/result/sentiment_eval/endding_gpt2_origin_cls.txt'
#file_ref = './endding_tcvae.ref'
#file_ref = './sentiment_pplm_ref.txt'
file_ref = '/home/ubuntu/birdring/SSAP/result/sentiment_eval/sentiment_predict_ref.txt'

list_ending = []
list_ref = []

with open(file_ending, "r", encoding="utf-8") as f:
    for line in f:
        list_ending.append(line)

i = 0
cnt = 0
final = 0

with open(file_ref,"r",encoding="utf-8") as f:
    for line in f:
        #print(line, list_ending[i])
        if(line == list_ending[i]) : cnt += 1
        i += 1
        
print(i)
print(cnt)
print(cnt / i)
