import os
import glob
import nltk
 
'''
path = "./genia/"
files = glob.glob(os.path.join(path, "*.dep"))
output_path = "./dataset/"

f_tok = open(output_path+"geniaquarter.tok", "w", encoding="utf-8")
f_dep = open(output_path+"geniaquarter.dep", "w", encoding="utf-8")

idx = 0

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            a = line.strip().split("(")
            if (len(a) < 2):
                f_dep.write("\n")
                continue
            b = a[1].split(",")
            c = ",".join(b[1:])
            f_dep.write(a[0]+" "+b[0]+" "+c.strip()[:-1]+"\n")

    file_path = file.split(".")
    file_path[-1] = "morph"
    with open((".").join(file_path), "r", encoding="utf-8") as f:
        for line in f.readlines():
            f_tok.writelines(line)
    
    idx += 1
    if (idx == 50):
        break
'''


'''

path = "./dataset/questions.txt"
snowballStemmer = nltk.stem.SnowballStemmer('english')

output_path = "./dataset/"
f_tok = open(output_path+"question.tok", "w", encoding="utf-8")
#f_merge = open(output_path+"question.cleanmerge", "w", encoding="utf-8")

cnt = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        a = line.strip().split(" ")
        a[-1] = a[-1][:-1]
        if (a[-1] != '?'):
            a.append('?')
        for word in a:
            f_tok.writelines(snowballStemmer.stem(word)+"\n")
            if (word == '?'):
                cnt += 1
                f_tok.writelines("\n")
        #f_merge.writelines((" ").join(a)+"\n")

print(cnt)
'''

path = "./dataset/genia_origin.dep"
f_dep = open("./dataset/genia.dep", "w", encoding="utf-8")
with open(path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        a = line.strip().split("(")
        if (len(a) < 2):
            f_dep.write("\n")
            continue
        if (a[0] == "punct"):
            continue
        a[1] = "(".join(a[1:])
        b = a[1].split(", ")
        c = ", ".join(b[1:])
        f_dep.write(a[0]+" "+b[0]+" "+c.strip()[:-1]+"\n")
       