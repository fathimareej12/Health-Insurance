from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def apply_undersampling(X_train,y_train):
    print('Class Distribution before undersampling')
    print(Counter(y_train))

    rus=RandomUnderSampler(random_state=42)
    X_train,y_train=rus.fit_resample(X_train,y_train)

    print('Class Distribution after undersampling')
    print(Counter(y_train))

    return (X_train,y_train  )