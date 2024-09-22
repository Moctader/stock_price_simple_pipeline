#import funcs.misc as misc
import time, random
import tensorflow as tf 
import pandas as pd
from typing import List, Text
import logging
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetSummaryMetric, ColumnDriftMetric, RegressionQualityMetric
from evidently.report import Report
import pendulum
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from typing import Any  
import yaml
import time
import random
import logging
import pandas as pd
import numpy as np
import pandas as pd
from typing import Any
import yaml
import os
import logging
import datetime
from typing import List, Text, Dict

# REMEMBER TO TRACK WHAT PIP LIBRARIES YOU NEED TO HAVE INSTALLED THIS CODE TO WORK !!!
# REMEMBER TO TRACK WHAT PIP LIBRARIES YOU NEED TO HAVE INSTALLED THIS CODE TO WORK !!!
# REMEMBER TO TRACK WHAT PIP LIBRARIES YOU NEED TO HAVE INSTALLED THIS CODE TO WORK !!!

root_path = '/Users/moctader/TrustworthyAI/arcada_pipelines/03_pipeline'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class create_metric_suite:
    def __init__(self):
        # Initialize imputer and scaler
        self.imputer = None
        self.scaler = None
        self.model = None  # Placeholder for the model
        self.timesteps = 3  # Number of timesteps for LSTM input

        # REQUIRED YAML PARAMS
        self.REQUIRED_YAML_PARAMS = {
            'artificial_delay': float,
            'threshold': float
        }


    

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")



    def prepare_data(self, reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Keep 'adjusted_close' in reference and current data
        reference = reference[['open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
        current = current[['open', 'high', 'low', 'close', 'adjusted_close', 'volume']]

        # Step 2: Imputation and Scaling
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        # Impute and scale the data for both reference and current datasets
        reference_imputed = self.imputer.fit_transform(reference[['open', 'high', 'low', 'close', 'volume']])
        reference_scaled = self.scaler.fit_transform(reference_imputed)
        current_imputed = self.imputer.transform(current[['open', 'high', 'low', 'close', 'volume']])
        current_scaled = self.scaler.transform(current_imputed)

        # Step 3: Convert the scaled data back into DataFrames
        reference_df_scaled = pd.DataFrame(reference_scaled, columns=['open', 'high', 'low', 'close', 'volume'])
        current_df_scaled = pd.DataFrame(current_scaled, columns=['open', 'high', 'low', 'close', 'volume'])

        # Step 4: Predict and store the predictions using the DataFrame format
        reference_predictions = self.predict_and_store(reference_df_scaled)
        current_predictions = self.predict_and_store(current_df_scaled)

        # Step 5: Add predictions back into the original DataFrame
        reference['predictions'] = reference_predictions
        current['predictions'] = current_predictions

        # Step 6: Save to Parquet format
        os.makedirs('data', exist_ok=True)
        reference.to_parquet('data/reference_data.parquet')
        current.to_parquet('data/current_data.parquet')

        print('Predictions saved to parquet files: data/reference_data.parquet and data/current_data.parquet')
       
        # Return the updated DataFrames
        return reference, current
    


    def predict_and_store(self, data: pd.DataFrame) -> pd.Series:
        predictions = []
        for i, row in data.iterrows():  # Now data is correctly a DataFrame, so iterrows works
            input_df = pd.DataFrame([row])

            # Apply imputation and scaling again if needed (though it's already done)
            input_df = self.imputer.transform(input_df)
            input_df = self.scaler.transform(input_df)

            # Handle sequence creation for your timesteps (assuming self.timesteps is defined)
            repeated_input = np.tile(input_df, (self.timesteps, 1)) if input_df.shape[0] < self.timesteps else input_df[-self.timesteps:]
            input_sequence = np.reshape(repeated_input, (1, self.timesteps, input_df.shape[1]))

            # Make prediction using your model
            prediction = self.model.predict(input_sequence)
            predictions.append(prediction.flatten()[0])
        
        # Return predictions as a pandas Series
        return pd.Series(predictions)


    
    
    def generate_reports(self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        num_features: List[Text],
        cat_features: List[Text],
        prediction_col: Text,
        target_col: Text,
        #timestamp: float,
        ) -> None:
            
        logging.info("Prepare column_mapping object for Evidently reports")
        column_mapping = ColumnMapping()
        column_mapping.target = target_col
        column_mapping.prediction = prediction_col
        column_mapping.numerical_features = num_features
        column_mapping.categorical_features = cat_features


        logging.info("Create a model performance report")
        model_performance_report = Report(metrics=[RegressionQualityMetric()])
        model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
        )

        logging.info("Target drift report")
    
        target_drift_report = Report(metrics=[ColumnDriftMetric(target_col)])
        target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
        )

        logging.info("Save metrics to database")
        model_performance_report_content: Dict = model_performance_report.as_dict()
        target_drift_report_content: Dict = target_drift_report.as_dict()


        logging.info("Commit data metrics to database")
        self.commit_data_metrics_to_db1(
            model_performance_report=model_performance_report_content,
            target_drift_report=target_drift_report_content,
            #timestamp=timestamp,
        )



    
    def model_performance_and_Target_drift_analysis(self, current_data,reference_data):

        target_col = "shifted_close"
        num_features = ['open', 'high', 'low', 'close']
        cat_features = []
        prediction_col = "predicted"
    
        # Load data
        # current_data = pd.read_parquet("/Users/moctader/TrustworthyAI/arcada_pipelines/03_pipeline/machine_learning/model_analysis/data/current_data.parquet")
        # reference_data = pd.read_parquet("/Users/moctader/TrustworthyAI/arcada_pipelines/03_pipeline/machine_learning/model_analysis/data/reference_data.parquet")


        if current_data.shape[0] == 0:
            print("Current data is empty!")
            print("Skip model monitoring")

        else:

            self.generate_reports(
                current_data=current_data,
                reference_data=reference_data,
                num_features=num_features,
                cat_features=cat_features,
                prediction_col=prediction_col,
                target_col=target_col,
                #timestamp=pendulum.parse(ts).timestamp(),
            )
        


    def numpy_to_standard_types(self, input_data: Dict) -> Dict:
        """Convert numpy type values to standard Python types in flat(!) dictionary.

        Args:
            input_data (Dict): Input data (flat dictionary).

        Returns:
            Dict: Dictionary with standard value types.
        """

        output_data: Dict = {}

        for k, v in input_data.items():
            if isinstance(v, np.generic):
                v = v.item()
            output_data[k] = v

        return output_data
    
 
    
    def parse_target_drift_report(self, target_drift_report: Dict) -> Dict:

        assert len(target_drift_report["metrics"]) == 1
        drift_metric: Dict = target_drift_report["metrics"][0]
        assert drift_metric["metric"] == "ColumnDriftMetric"
        raw_drift_metric_result: Dict = drift_metric["result"]
        drift_metric_result: Dict = {
            k: v
            for k, v in raw_drift_metric_result.items()
            if isinstance(v, (int, float, str, np.generic))
        }
        drift_metric_result = self.numpy_to_standard_types(drift_metric_result)
        remove_fields: List[Text] = ["column_name", "column_type"]

        for field in remove_fields:
            del drift_metric_result[field]
        print(drift_metric_result)



        drift_thresholds = 0.02
        if drift_metric_result['drift_score'] > drift_thresholds:
            logger.info("Model drift detected: Drift score exceeds threshold")

        if drift_metric_result['drift_detected'] != drift_thresholds:
            logger.info("Model drift detected: Drift detected status does not match threshold")

        return drift_metric_result
    


    def parse_model_performance_report(self, model_performance_report: Dict) -> Dict:

        assert len(model_performance_report["metrics"]) == 1
        quality_metric: Dict = model_performance_report["metrics"][0]
        assert quality_metric["metric"] == "RegressionQualityMetric"
        raw_quality_metric_result: Dict = quality_metric["result"]
        current_metrics: Dict = raw_quality_metric_result['current']
        raw_quality_metric_result.update(current_metrics)
        quality_metric_result: Dict = {
            k: v
            for k, v in raw_quality_metric_result.items()
            if isinstance(v, (int, float, str, np.generic))
        }
        quality_metric_result = self.numpy_to_standard_types(quality_metric_result)
        # Check if any metric exceeds its threshold

        #performance_thresholds = self.selected_model['performance_thresholds']
        performance_thresholds = 0.01

        if quality_metric_result['rmse'] > performance_thresholds:
            logger.info("Model performance drift detected: RMSE exceeds threshold")

        if quality_metric_result['mean_abs_error'] > performance_thresholds:
            logger.info("Model performance drift detected: Mean Absolute Error exceeds threshold")

        if quality_metric_result['r2_score'] < performance_thresholds:
            logger.info("Model performance drift detected: R2 Score is below threshold")

        return quality_metric_result

  

    def commit_data_metrics_to_db1(self, model_performance_report: Dict, target_drift_report: Dict) -> None:

        ### MODEL PERFORMANCE REPORT
        model_performance_report = self.parse_model_performance_report(model_performance_report)
        model_performance_report['id'] = datetime.datetime.now()
        #insert_model_performance(model_performance_report)


        ## TARGET DRIFT REPORT
        target_drift_report = self.parse_target_drift_report(target_drift_report)
        target_drift_report['id'] = datetime.datetime.now()
        #insert_target_drift(target_drift_report)
    
########################################################################################################
########################################################################################################








#if __name__=='__main__':
suite = create_metric_suite()
#     suite.load_model(f'{root_path}/machine_learning/model_suites/model_lstm_v1.keras')
#     current_data = pd.read_csv(f'{root_path}/datasets/finance_fresh.csv')
#     reference_data = pd.read_csv(f'{root_path}/datasets/finance_historical.csv')   
#     #ref, curr = suite.prepare_data(current_data, reference_data)
#     # print("Reference Data Head:")
#     # print("Current Data Head:")
#     # print(curr.head())
#     suite.model_performance_and_Target_drift_analysis(current_data,reference_data)
   