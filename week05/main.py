import numpy as np

def normalization(data):
    """
    Normalize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the normalized data on each column separately
    """
    return None

def standardization(data):
    """
    Standardize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the standardized data on each column separately
    """
    return None

def binning(data, n_bin=5):
    """
    Discretize the data using equal-width intervals and return bin identifier 
    encoded as an integer value (from 0 to n_bin-1).
    Input:  data = a 1d numpy array.
    Output:  the discretized data and a 1d arrat if bin edges
    """
    return None, None

def label_encoding(data):
    """
    Label encoding the given categorial data in the alphabetical order
    Input:  data = a 1d numpy array of str.
    Output:  the 1d array of encoded labels and a 1d array of class label
    """
    return None, None

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1,2,3))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    scaler = MinMaxScaler()
    skl = scaler.fit_transform(data[:,0:1])
    our = normalization(data[:,0:1])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    scaler = StandardScaler()
    skl = scaler.fit_transform(data[:,1:3])
    our = standardization(data[:,1:3])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    le = LabelEncoder()
    skl = le.fit_transform(target)
    our, cls = label_encoding(target)
    print(skl[-5:])
    print(our[-5:])
    print(le.classes_)
    print(cls)

    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    skl = est.fit_transform(data[:,1:2])
    our, e = binning(data[:,1:2])
    print(skl[-5:,:])
    print(est.n_bins_, est.bin_edges_)
    print(our[-5:,:])
    print(e)
    assert np.allclose(skl, our)
    assert np.allclose(est.bin_edges_[0], e)