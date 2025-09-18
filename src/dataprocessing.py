from utils.logger import logging
from utils.custom_exception import CustomException
import sys
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
import os
import joblib


class DataPreProcessing:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.x = None
        self.y = None
        self.mappings = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  
        os.makedirs(self.output_path, exist_ok=True)
        

    def load_data(self) -> pd.DataFrame:
        try:
            self.data = pd.read_csv(self.input_path)
            logging.info(f"Data loaded successfully from {self.input_path}")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data from {self.input_path}")
            raise CustomException(e, sys)

    def preprocess_data(self):
        try:
            self.data = self.data.drop('Patient_ID', axis=1)
            self.x = self.data.drop('Survival_Prediction', axis=1)
            self.y = self.data['Survival_Prediction']

            cat_cols = [i for i in self.x.columns if self.x[i].dtype == 'object']

            for col in cat_cols:
                self.x[col] = self.label_encoder.fit_transform(self.x[col])
                self.mappings[col] = {label: code for label, code in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))}

            logging.info("Categorical columns encoded successfully")
            logging.info("Data preprocessing completed")

        except Exception as e:
            logging.error("Error in data preprocessing")
            raise CustomException(e, sys)
    
    def feature_selection(self):
        try:
            x_train, _, y_train, _ = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
            x_cat = x_train.select_dtypes(['int64', 'float64'])
            selector = SelectKBest(score_func=chi2, k=8)
            x_cat = selector.fit(x_cat, y_train)

            top_features = x_cat.get_support(indices=True)
            self.x = self.x.iloc[:, top_features]
            logging.info(f"Feature selection completed and the features are {self.x.columns.tolist()}")
                        
        except Exception as e:
            logging.error("Error in feature selection")
            raise CustomException(e, sys)
    
    def split_scale_data(self):
        try:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42, stratify = self.y)
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            logging.info("Data splitting and scaling completed")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Error in splitting and scaling data")
            raise CustomException(e, sys)

    def save_artifacts(self):
        try:
            joblib.dump(self.x_train, os.path.join(self.output_path, 'x_train.pkl'))
            joblib.dump(self.y_train, os.path.join(self.output_path, 'y_train.pkl'))
            joblib.dump(self.x_test, os.path.join(self.output_path, 'x_test.pkl'))
            joblib.dump(self.y_test, os.path.join(self.output_path, 'y_test.pkl'))
            logging.info(f"Artifacts saved successfully in {self.output_path}")
        except Exception as e:
            logging.error("Error saving artifacts")
            raise CustomException(e, sys)
    
    def run(self):
        try:
            self.load_data()
            self.preprocess_data()
            self.feature_selection()
            self.x_train, self.x_test, self.y_train, self.y_test = self.split_scale_data()
            self.save_artifacts()
            logging.info("Data preprocessing pipeline completed successfully")
        except Exception as e:
            logging.error("Error in running data preprocessing pipeline")
            raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     input_path = "artifacts/raw/colorectal_cancer_dataset.csv"
#     output_path = 'artifacts/processed/'
#     data_processor = DataPreProcessing(input_path, output_path)
#     data_processor.run()
