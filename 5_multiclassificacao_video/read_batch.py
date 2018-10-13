
import numpy as np

current_index_train_batch = 0
max_index_train_batch = 205

current_index_test_batch = 0
max_index_test_batch = 27

current_index_val_batch = 28
max_index_val_batch = 56

def get_next_train_batch():
    global current_index_train_batch, max_index_train_batch

    if current_index_train_batch > max_index_train_batch:
        current_index_train_batch = 0
        return -1, None, None

    file = "dataset/train_"+str(current_index_train_batch)+".csv"
    file = open(file, "r")
    lines = file.readlines()
    num_lines = len(lines)
    labels = np.ndarray(shape=(num_lines,3862), dtype=float)
    features = np.ndarray(shape=(num_lines,1152), dtype=float)

    for index, line in enumerate(lines):
        data = line.split(',')

        labels[index] = np.asarray(list(map(float, data[:3862])))
        features[index] = np.asarray(list(map(float, data[3862:-1])))

    file.close() 
    current_index_train_batch += 1

    return current_index_train_batch, labels, features

def get_next_val_batch():
    global current_index_val_batch, max_index_val_batch

    if current_index_val_batch > max_index_val_batch:
        current_index_val_batch = 28
        return -1, None, None

    file = "dataset/val_"+str(current_index_val_batch)+".csv"
    file = open(file, "r")
    lines = file.readlines()
    num_lines = len(lines)
    labels = np.ndarray(shape=(num_lines,3862), dtype=float)
    features = np.ndarray(shape=(num_lines,1152), dtype=float)

    for index, line in enumerate(lines):
        data = line.split(',')

        labels[index] = np.asarray(list(map(float, data[:3862])))
        features[index] = np.asarray(list(map(float, data[3862:-1])))

    file.close() 
    current_index_val_batch += 1

    return current_index_val_batch, labels, features    

def get_next_test_batch():
    global current_index_test_batch, max_index_test_batch

    if current_index_test_batch > max_index_test_batch:
        current_index_test_batch = 0
        return -1, None, None

    file = "../dataset/val_"+str(current_index_test_batch)+".csv"
    file = open(file, "r")
    lines = file.readlines()
    num_lines = len(lines)
    labels = np.ndarray(shape=(num_lines,3862), dtype=float)
    features = np.ndarray(shape=(num_lines,1152), dtype=float)

    for index, line in enumerate(lines):
        data = line.split(',')

        labels[index] = np.asarray(list(map(float, data[:3862])))
        features[index] = np.asarray(list(map(float, data[3862:-1])))

    file.close() 
    current_index_test_batch += 1

    return current_index_test_batch, labels, features  



meta_table = None

def show_meta_data(label_id):
    global meta_table
    if meta_table is None:
        file = open("yt8m_voc.csv", "r")
        meta_table = file.readlines()
        file.close()
        
    data = meta_table[label_id+1].split(",")
    
    print("- label id:", data[0], "name:", data[3], "wiki:", data[4])
    
    


    