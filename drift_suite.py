#import funcs.misc as misc
import tensorflow as tf 
import pandas as pd
from typing import List, Text
import logging
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetSummaryMetric, ColumnDriftMetric, RegressionQualityMetric
from evidently.report import Report
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Any  
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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class create_metric_suite:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.model = None  
        self.timesteps = 3  


    
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
    

suite = create_metric_suite()
