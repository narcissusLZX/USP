#fout = open("test.out", "w", encoding="utf-8")
flag = False
lst = ""
cnt = 0

with open("./test.out", "r", encoding="utf-8") as f:
    for line in f.readlines():

        '''
        a = line.strip().split("<")
        print(a)
        if (len(a) <= 1):
            continue
        elif (len(a) == 2):
            print(line, a)
            fout.write("Q:"+a[1].split("=")[-1][1:-2]+"\n")
        type, outstr = a[1].split(">")
        if (type == "answer"):
            fout.write("A:"+outstr+"\n")
            if (flag):
                gold = outstr
            else:
                gold = "wrong"
            fout.write("Gold:"+gold+"\n\n")
        elif (type=="label"):
            if(outstr == "correct"):
                flag = True
            else:
                flag = False
                
        '''
        a = line.strip().split(":")
        if (a[0] == 'Q'):
            if (a[1] == lst):
                continue
            else:
                print(a[1])
                cnt += 1
                lst = a[1]
print(cnt)