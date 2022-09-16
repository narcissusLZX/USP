

class Argument():
    def __init__(self, dataset, father, son, argType="None", idx=-1):
        self.dataset = dataset
        self.father = father
        self.son = son
        self.argType = argType
        if (idx == -1):
            idx = dataset.newArgumentIdx(self)
        self.idx = idx

    def store(self):
        data = {}
        data["father"] = self.father.idx
        data["son"] = self.son.idx
        data["argType"] = self.argType
        data["idx"] = self.idx
        return data

    def load(self, data):
        self.father = self.dataset.idx2occ[int(data["father"])]
        self.son = self.dataset.idx2occ[int(data["son"])]
        self.argType = data["argType"]
