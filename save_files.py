import _pickle as pickle

def save(nn_weights, name):
    file = open(name, "wb")
    pickle.dump(nn_weights, file)
    file.close()

def load(name):
    file = open(name, "rb")
    data = pickle.load(file)
    file.close()
    return data