import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from utils.custom_exception import CustomException
import sys
from utils.logger import logging
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='nileshnandan.ts', repo_name='Colorectal-cancer-prediction', mlflow=True)

class ModelTraining:
    def __init__(self, model_path: str ="artifacts/model/" , processed_data_dir: str = "artifacts/processed/"):
        self.model_path = model_path
        self.processed_data_dir = processed_data_dir
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.processed_data_dir, 'x_train.pkl'))
            self.x_test = joblib.load(os.path.join(self.processed_data_dir, 'x_test.pkl'))
            self.y_train = joblib.load(os.path.join(self.processed_data_dir, 'y_train.pkl'))
            self.y_test = joblib.load(os.path.join(self.processed_data_dir, 'y_test.pkl'))
            logging.info("Processed data loaded successfully")
    
        except Exception as e:
            logging.error("Error in loading processed data")
            raise CustomException(e, sys)
        
    def train_model(self):
        try:
            model = GradientBoostingClassifier()
            model.fit(self.x_train, self.y_train)
            logging.info("Model training completed")
            return model
        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

    def evaluate_model(self, model):
        try:
            y_pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            y_proba = model.predict_proba(self.x_test)[:, 1] if (len(self.y_test.unique())==2) else None
            roc_auc = roc_auc_score(self.y_test, y_proba)

            logging.info(f"Model evaluation metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}, ROC-AUC={roc_auc}")
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
        except Exception as e:
            logging.error("Error in model evaluation")
            raise CustomException(e, sys)

    def save_model(self, model):
        try:
            joblib.dump(model, os.path.join(self.model_path, 'model.pkl'))
            logging.info(f"Model saved at {os.path.join(self.model_path, 'model.pkl')}")
        except Exception as e:
            logging.error("Error in saving the model")
            raise CustomException(e, sys)
    
    def run(self):
        try:
            with mlflow.start_run():
                self.load_data()
                model = self.train_model()
                d = self.evaluate_model(model)
                self.save_model(model)

                mlflow.log_artifact(os.path.join(self.model_path, 'model.pkl'), "model.pkl")

                mlflow.log_artifacts(self.processed_data_dir)

                mlflow.log_metrics(d)

                # mlflow.sklearn.log_model(model, "GradientBoostingClassifier_Cancer_Prediction")
            logging.info("Model training pipeline completed successfully")
        except Exception as e:
            logging.error("Error in running model training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    model_trainer = ModelTraining()
    model_trainer.run()