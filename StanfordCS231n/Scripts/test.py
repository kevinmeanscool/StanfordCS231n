import numpy as np

# take first 1000 for validation
Xval_rows = np.array([[1,2,3,4,5]])
print(Xval_rows)
print(np.shape(Xval_rows))
Yval = np.array([4])
# keep last 49,000 for train
Xtr_rows = np.array([[1,2,3,3,5],[1,3,3,3,5],[2,2,3,3,5],[1,2,3,4,5]])
print(Xtr_rows)
Ytr = np.array([[1],[2],[3],[4]])

for i in range(Xtr_rows.shape[0]):
    distances = np.sqrt(np.sum(np.square(Xval_rows-Xtr_rows[i,:]), axis = 1))
    print(distances)


