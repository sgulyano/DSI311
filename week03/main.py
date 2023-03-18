import numpy as np

def find_accuracy(y_true, y_pred):
    """
    Evaluate the accuracy of the prediction
    Input:  y_true = ground truth label as a 1d-array of int (class label).
            prediciton = prediction as a 1d-array of int (class label).
    Output:  accuracy of all samples
    """
    return None

def find_precision(y_true, y_pred):
    """
    Evaluate the precision of the prediction for each class.
    Input:  y_true = ground truth label as a 1d-array of int (class label).
            prediciton = prediction as a 1d-array of int (class label).
    Output:  precision of each class
    """
    return None

def find_recall(y_true, y_pred):
    """
    Evaluate the recall of the prediction for each class.
    Input:  y_true = ground truth label as a 1d-array of int (class label).
            prediciton = prediction as a 1d-array of int (class label).
    Output:  recall of each class
    """
    return None

def find_f1score(y_true, y_pred):
    """
    Evaluate the F1-score of the prediction for each class.
    Input:  y_true = ground truth label as a 1d-array of int (class label).
            prediciton = prediction as a 1d-array of int (class label).
    Output:  F1-score of each class
    """
    return None


if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    # convert target (in str) to number (int)
    le = LabelEncoder()
    y = le.fit_transform(target)
    print(y, type(y.dtype), le.classes_)

    # split data to create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # build logistic regression model and make prediction
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred, y_test)

    # evaluate 
    print(f'Accuracy = {accuracy_score(y_test, y_pred):.3f}')
    print(f'Accuracy = {find_accuracy(y_test, y_pred):.3f}')
    print(f'Precision = \n{precision_score(y_test, y_pred, average=None)}')
    print(f'Precision = \n{find_precision(y_test, y_pred)}')
    print(f'Recall = \n{recall_score(y_test, y_pred, average=None)}')
    print(f'Recall = \n{find_recall(y_test, y_pred)}')
    print(f'F1-score = \n{f1_score(y_test, y_pred, average=None)}')
    print(f'F1-score = \n{find_f1score(y_test, y_pred)}')
