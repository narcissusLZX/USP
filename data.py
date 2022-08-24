path = "/data/genia.txt"

with open(path, "r") as f:
    for line in f:
        a = line.strip().split(" ")
        