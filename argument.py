

class Argument():
    def __init__(self, dataset, father, son, argType="None", idx=-1):
        self.dataset = dataset
        self.father = father
        self.son = son
        self.argType = argType
        if (idx == -1):
            idx = dataset.newArgumentIdx(self)
        self.idx = idx
