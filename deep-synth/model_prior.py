from data import RenderedScene, ObjectCategories
import os
import pickle
import numpy as np
import utils

"""
Simple bigram model
Very ugly piece of code
But well it kinda works
"""
#I forgot how I implemented this, so sorry about lack of documentation -.-
class ModelPrior():

    def __init__(self):
        pass

    def learn(self, data_folder, data_root_dir=None):
        if data_root_dir is None:
            data_root_dir = utils.get_data_root_dir()
        data_dir = f"{data_root_dir}/{data_folder}"
        self.data_dir = data_dir
        self.category_map = ObjectCategories()

        files = os.listdir(data_dir)
        files = [f for f in files if ".pkl" in f and not "domain" in f]

        with open(f"{data_dir}/final_categories_frequency", "r") as f:
            lines = f.readlines()
            cats = [line.split()[0] for line in lines]

        self.categories = [cat for cat in cats if cat not in set(['window', 'door'])]
        self.cat_to_index = {self.categories[i]:i for i in range(len(self.categories))}

        with open(f"{data_dir}/model_frequency", "r") as f:
            lines = f.readlines()
            models = [line.split()[0] for line in lines]
            self.model_freq = [int(l[:-1].split()[1]) for l in lines]

        self.models = [model for model in models if self.category_map.get_final_category(model) not in set(['window', 'door'])]
        self.model_to_index = {self.models[i]:i for i in range(len(self.models))}

        N = len(self.models)
        self.num_categories = len(self.categories)
        
        self.model_index_to_cat = [self.cat_to_index[self.category_map.get_final_category(self.models[i])] for i in range(N)]

        self.count = [[0 for i in range(N)] for j in range(N)]

        for index in range(len(files)):
        #for index in range(100):
            with open(f"{data_dir}/{index}.pkl", "rb") as f:
                (_, _, nodes), _ = pickle.load(f)
            
            object_nodes = []
            for node in nodes:
                modelId = node["modelId"]
                category = self.category_map.get_final_category(modelId)
                if not category in ["door", "window"]:
                    object_nodes.append(node)

            for i in range(len(object_nodes)):
                for j in range(i+1, len(object_nodes)):
                    a = self.model_to_index[object_nodes[i]["modelId"]]
                    b = self.model_to_index[object_nodes[j]["modelId"]]
                    self.count[a][b] += 1
                    self.count[b][a] += 1
            print(index, end="\r")

        self.N = N

    def save(self, dest=None):
        if dest == None:
            dest = f"{self.data_dir}/model_prior.pkl"
        with open(dest, "wb") as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, data_dir):
        source = f"{data_dir}/model_prior.pkl"
        with open(source, "rb") as f:
            self.__dict__ = pickle.load(f)

    
    def sample(self, category, models):
        N = self.N
        indices = [i for i in range(N) if self.model_index_to_cat[i]==category]
        p = [self.model_freq[indices[i]] for i in range(len(indices))]
        p = np.asarray(p)
        for model in models:
            i = self.model_to_index[model]
            p1 = [self.count[indices[j]][i] for j in range(len(indices))]
            p1 = np.asarray(p1)
            p1 = p1/p1.sum()
            p = p * p1
        
        p = p/sum(p)
        numbers = np.asarray([i for i in range(len(indices))])
        return self.models[indices[np.random.choice(numbers, p=p)]]

    def get_models(self, category, important, others):
        N = self.N
        indices = [i for i in range(N) if self.model_index_to_cat[i]==category]
        to_remove = []

        freq = [self.model_freq[indices[i]] for i in range(len(indices))]
        total_freq = sum(freq)
        for j in range(len(indices)):
            if freq[j] / total_freq < 0.01:
                if not indices[j] in to_remove:
                    to_remove.append(indices[j])

        for model in important:
            i = self.model_to_index[model]
            freq = [self.count[indices[j]][i] for j in range(len(indices))]
            total_freq = sum(freq)
            if total_freq > 0:
                for j in range(len(indices)):
                    if freq[j] / total_freq < 0.1:
                        if not indices[j] in to_remove:
                            to_remove.append(indices[j])

        for model in others:
            i = self.model_to_index[model]
            freq = [self.count[indices[j]][i] for j in range(len(indices))]
            total_freq = sum(freq)
            if total_freq > 0:
                for j in range(len(indices)):
                    if freq[j] / total_freq < 0.05:
                        if not indices[j] in to_remove:
                            to_remove.append(indices[j])
        
        for item in to_remove:
            if len(indices) > 1:
                indices.remove(item)
        
        return [self.models[index] for index in indices]

            
if __name__ == "__main__":
    a = ModelPrior()
    a.learn("bedroom")
    a.save()
    #a.load()
    #print(a.get_models(1, ["415"]))
