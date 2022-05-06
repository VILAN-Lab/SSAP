import os


file = './grf.txt'
file_2 = './grf2.txt'

file_data = ""

with open(file, "r", encoding="utf-8") as f:
    for line in f:
        line = line[1:]
        line = line.lower()
        a = line[-2]
        line = line[:-2] + ' ' + a + '\n'
        #line = line.split('generate:---->')[1] 
        file_data += line 

#print(file_data)

with open(file_2,"w",encoding="utf-8") as f:
        f.write(file_data)

