
from sklearn.model_selection import train_test_split

''' SPLIT THE DATASET '''

# NORMAL SPLITTING: TRAINING, VAL AND TEST SETS
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=13)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# SPLITTING WITHOUT TEST SET, BUT ONLY WITH TRAINING AND VALIDATION SETS
def split_dataset_without_test(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=13, shuffle=True)

    return (X_train, y_train), (X_val, y_val)