import joblib
import pandas as pd
from config.paths import MODEL_PATH

class pred_pipeline:
    def __init__(self):
        self.model=joblib.load(MODEL_PATH/'GaussianNB.pkl')
        self.preprocessor=joblib.load(MODEL_PATH/'preprocessor.pkl')

    def predict(self,insurance_data=pd.DataFrame):
        transformed_data = self.preprocessor.transform(insurance_data)
        prediction=self.model.predict(insurance_data)[0]

        probability = self.model.predict_proba(transformed_data)[0]
        return prediction,probability

def main():
    insurance={
        "Gender": "Male",

        "Age": 44,

        "Drving_license": 1,

        "Region_code": 28.0,

        "previously_Insured": 0,

        "Vehicle_Age": ">2 years",

        "Vehicle_Damage": "yes",

        "Annual_Premium": 40454.0,

        "Policy_Sales_Channel": 26.0,

        "Vintage": 217,
    }

    Insurance_df=pd.Datframe(insurance)
    pipe=pred_pipeline()
    prediction,probability=pipe.predict(insurance_df)
    if prediction==1:
        print('client interest in vehicle Insurance')
        print(f'Probability:{probability[1]*100:.2f}%')

    else:
        print('client not interest in vehicle Insurance')
        print(f'Probability: {probability[0]*100:.2f}%')




