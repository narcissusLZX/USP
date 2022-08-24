from nltk.parse.stanford import StanfordParser
import nltk
import os

if __name__ == '__main__':
    nltk.internals.config_java('E:\java\bin\java.exe')
    java_path = "E:\java/"
    os.environ['JAVAHOME'] = java_path
    
    stanford_parser_dir = r"C:\Users\lzx\Downloads\stanford-parser-4.2.0\stanford-parser-full-2020-11-17/"
    eng_model_path = r"C:\Users\lzx\Downloads\stanford-parser-4.2.0\stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    my_path_to_models_jar = stanford_parser_dir + "stanford-parser-4.2.0-models.jar"
    my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

'''
    parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar,
                            path_to_jar=my_path_to_jar)

    s = list(parser.parse(
        "The President of the United States is Trump".split()))
    # s = parser.raw_parse(
    #     "the quick brown fox jumps over the lazy dog")
    for line in s:
        print("line: ")
        line.draw()
        print(line)
'''

from nltk.parse.stanford import StanfordDependencyParser

eng_parser = StanfordDependencyParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar,
                                      path_to_jar=my_path_to_jar)

result = eng_parser.parse("I can live on air and on land.".split(" "))
#result = eng_parser.parse("Ravens defeated the Pittsburgh team .".split(" "))
dep = result.__next__()

#print(list(dep.triples()))

def dfs(deps, addr, fa):
    dep = deps.get_by_address(addr)
    if (addr == 0):
        cur = None
    else:
        cur = dep['word']+"-"+str(dep['address'])
    if (fa != None):
        print(dep['rel']+" "+fa+" "+cur)
    for d in dep['deps']:
        for addr2 in dep['deps'][d]:
            dfs(deps, addr2, cur)

dfs(dep, 0, None)