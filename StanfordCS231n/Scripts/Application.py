import numpy as np

import heapq as heapq

"Data Process"
 # read batches and return type Dictionary
 # Dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10(file):
    dict_1 = unpickle(file+'\\data_batch_1')
    dict_2 = unpickle(file+'\\data_batch_2')
    dict_3 = unpickle(file+'\\data_batch_3')
    dict_4 = unpickle(file+'\\data_batch_4')
    dict_5 = unpickle(file+'\\data_batch_5')
    # Xtr(of size 50,000 x 32 x 32 x 3) holds all the images in the training set
    # Ytr (of length 50,000) holds the training labels
    # But new CIFAR10 already implements that transform part of Xtr&Ytr to Xte&Yte
    # So merging them is enough and no more reshape
    Xtrain = np.r_[dict_1[b'data'],dict_2[b'data'],dict_3[b'data'],dict_4[b'data'],dict_5[b'data']]
    Ytrain = dict_1[b'labels']+dict_2[b'labels']+dict_3[b'labels']+dict_4[b'labels']+dict_5[b'labels']
    # Get test_batch
    test_dict = unpickle(file+'\\test_batch')
    Xtest = test_dict[b'data']
    Ytest = test_dict[b'labels']

    return Xtrain,Ytrain,Xtest,Ytest


"Classifier Train"

# Here is an implementation of a simple Nearest Neighbor classifier 
# with the L1 distance that satisfies this template:

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # X is training photos and Y is testing
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X , k):
    """ X is N x D where each row is an example we wish to predict label for """
    # X is photos under test

    # shape[0] will return length of first dimension
    print(X)
    num_test = X.shape[0]
    print( 'X:',X.shape[0])
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = type(self.ytr[0]))
    # create a list saving
     
    print('num_test:%d' %(num_test))
    # loop over all test rows
    # range(x) can create a ingeter list between 0 and x for loop
    for i in range(num_test):
      print("k = %d ,process: %f / %d " %(k,i,num_test))
      # find the nearest training image to the i'th test image
      # using the L2 distance (sum of euclidean distance between two vectors)
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)).tolist()
      print(np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)))
      # heapq.nsmallest will get index(es) of the smallest k distance 
      # k_NearestNeighbor_index = list(map(distances.tolist().index,heapq.nsmallest(k,distances)))
      k_NearestNeighbor_index = []
      
      for j in range(k):
        crruent_minial = distances.index(min(distances))
        k_NearestNeighbor_index.append(crruent_minial)
        if j == 0 :
          knn_label_min = crruent_minial
        distances[crruent_minial] = max(distances)+1

      print(k_NearestNeighbor_index)
      # create a array for labels that has the same length as k_NearestNeighbor_index 
      k_NearestNeighbor_label = []
      for index in k_NearestNeighbor_index:
       print(index)
       k_NearestNeighbor_label.append(self.ytr[index]) 

      print(k_NearestNeighbor_label)
      # np.unique will return a array that array[0] denotes labels being deleted 
      # duplicated labels and arrary[1] is frequency of labels
      knn_freq = np.unique(k_NearestNeighbor_label, return_counts=True)

      ones = np.ones(len(knn_freq[1]))

      if (ones == knn_freq[1]).all() == True:
        knn_label = self.ytr[knn_label_min]
      else:
       print(knn_freq)
       # knn_freq_list is knn_freq in the form of list
       knn_freq_list = knn_freq[1].tolist()
       # get the label of highest frequency
       knn_label = knn_freq[0][knn_freq_list.index(max(knn_freq_list))] 
       # records the label to implement logcal proximity
       # predict the label of the nearest example

      print(knn_label)
      Ypred[i] = knn_label

    # return labels as results
    return Ypred


Xtr_rows , Ytr, Xte_rows , Yte = load_CIFAR10('E:\\pyDB\\cifar-10-batches-py')

# take first 1000 for validation
Xval_rows = np.array([[1,2,3,4,5]])
print(Xval_rows)
print(np.shape(Xval_rows))
Yval = np.array([4])
# keep last 49,000 for train
Xtr_rows = np.array([[1,2,3,3,5],[1,3,3,3,5],[2,2,3,3,5],[1,2,3,4,5]])
print(Xtr_rows)
Ytr = np.array([[1],[2],[3],[4]])

# find hyperparameters that work best on the validation set
validation_accuracies = []

print(np.shape(Xtr_rows),np.shape(Ytr),np.shape(Xte_rows),np.shape(Yte))

print(round(Xtr_rows.shape[0]/5))

for k in [4]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k)
  acc = np.mean(Yval_predict == Yval)
  print ('accuracy: %f' % (acc,))

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))

print(validation_accuracies)