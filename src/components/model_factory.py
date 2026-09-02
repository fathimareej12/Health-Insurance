from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def get_model():
    models={
        'Logistic Regression':
            LogisticRegression(random_state=42,
                               max_iter=1000),
        'Decision Tree':
        DecisionTreeClassifier(random_state=42),

        'Random Forest':
        RandomForestClassifier(random_state=42),

        'Gradient Boosting':
        GradientBoostingClassifier(random_state=42),

        'AdaBoost':
        AdaBoostClassifier(random_state=42),

        'K_Neighbors':
        KNeighborsClassifier(n_neighbors=5),

        'GaussianNB':
        GaussianNB()
    }

    return models

def get_main():
    model=get_model()
    print('Available Models:')
    for model_name in model:
        print(model_name)


if __name__ == '__main__':
    get_main()






