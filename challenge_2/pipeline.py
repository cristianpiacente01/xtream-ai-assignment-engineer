"""
Pipeline for processing, transforming, evaluating, and deploying a diamonds dataset model.
This script uses the Luigi library to manage data processing tasks in a sequential and dependent way.

Cristian Piacente
"""

import luigi
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
import os
import warnings
import logging


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('luigi-pipeline')

# Get configuration file
config = luigi.configuration.get_config()

# Dict that contains the default paths configurated in luigi.cfg
default_paths = {
    'input_csv': config.get('DataPreprocessing', 'input_csv'),
    'cleaned_csv': config.get('DataPreprocessing', 'cleaned_csv'),
    'transformed_csv': config.get('DataTransformation', 'transformed_csv'),
    'train_csv': config.get('SplitDataset', 'train_csv'),
    'test_csv': config.get('SplitDataset', 'test_csv'),
    'model_file': config.get('EvalModel', 'model_file'),
    'metrics_csv': config.get('PerformanceEval', 'metrics_csv'),
    'deploy_model_file': config.get('DeployModel', 'deploy_model_file')
}



class BetterCompleteCheck:
    """
    A mixin to enhance Luigi's Task with a robust complete method that considers input modifications.
    It ensures tasks are re-executed when their dependencies are updated.
    """

    # Luigi's complete method override, to know if a task doesn't need to be executed again
    def complete(self):
        # Added support for dict (also recursive)
        def to_list(obj):
            if type(obj) in (type(()), type([])):
                # For tuples and lists assume there's no recursion
                return obj
            elif type(obj) == type({}):
                # If dict, return the values list
                return_list = []
                for value in obj.values():
                    if type(value) == type({}):
                        # Dict as a dict value, recursion
                        for single_target in to_list(value):
                            return_list.append(single_target)
                    else:
                        # Else we assume a single target as the value
                        return_list.append(value)
                # List of single targets
                return return_list
            else:
                # We assume only 1 single target
                return [obj]

        # Get last modification in seconds (float value)
        def mtime(path):
            return os.path.getmtime(path)

        # Complete override logic

        # If an output doesn't exist, the task is not complete
        if not all(os.path.exists(out.path) for out in to_list(self.output())):
            return False

        # Consider the minimum last modification if multiple outputs
        self_mtime = min(mtime(out.path) for out in to_list(self.output()))

        for el in to_list(self.requires()):
            if not el.complete():
                # If an input is not complete, the task is not complete
                return False
            for output in to_list(el.output()):
                # If an input has an output that was modified, the task is not complete
                if mtime(output.path) > self_mtime:
                    return False

        # All checks passed, the task is complete
        return True



class DataPreprocessing(BetterCompleteCheck, luigi.Task):
    """
    Cleans the raw diamonds dataset by removing missing values, duplicates and inconsistent rows.
    Outputs a cleaned dataset file.

    Parameters:
    - input-csv: Path to the input csv file containing raw diamonds data. Default: 'datasets/diamonds/diamonds.csv'
    - cleaned-csv: Path to the output csv file for cleaned data. Default: 'datasets/diamonds/diamonds_cleaned.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    cleaned_csv = luigi.Parameter(default=default_paths['cleaned_csv'])
    

    def requires(self):
        # diamonds.csv is needed, use a fake task
        class FakeTask(luigi.Task):
            def output(_):
                return luigi.LocalTarget(self.input_csv)
        
        return FakeTask()
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read diamonds.csv
        df = pd.read_csv(self.input_csv)

        logger.info('Retrieved the raw dataset')

        # Drop rows with missing values
        df.dropna(inplace=True)

        logger.info('Dropped the missing values')

        # Drop duplicated rows
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)

        logger.info('Dropped the duplicates')

        # Possible categorical values
        possible_values = {
            'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
            'color': [chr(i) for i in range(ord('D'), ord('Z') + 1)], # From 'D' to 'Z'
            'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        }

        # Check numerical columns for negative or zero values
        numerical_checks = (df['carat'] <= 0) | (df['depth'] <= 0) | (df['table'] <= 0) | \
                        (df['price'] <= 0) | (df['x'] <= 0) | (df['y'] <= 0) | (df['z'] <= 0)

        # Check categorical columns for invalid values
        categorical_checks = (~df['cut'].isin(possible_values['cut'])) | \
                            (~df['color'].isin(possible_values['color'])) | \
                            (~df['clarity'].isin(possible_values['clarity']))

        # Combine checks to filter out inconsistent rows
        inconsistent_rows = numerical_checks | categorical_checks
        
        # Drop inconsistent rows
        df = df[~inconsistent_rows]

        logger.info('Dropped the inconsistent data')

        # Save to diamonds_cleaned.csv
        df.to_csv(self.output().path, index=False)

        logger.info('Saved to csv the cleaned dataset')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.cleaned_csv)
    


class DataTransformation(BetterCompleteCheck, luigi.Task):
    """
    Applies ordinal encoding to categorical features within the cleaned diamonds dataset.
    Outputs a dataset file with transformed features for model training.

    Parameters:
    - input-csv: Path to the input csv file containing raw diamonds data. Default: 'datasets/diamonds/diamonds.csv'
    - transformed-csv: Path for the output dataset with transformed features after preprocessing. Default: 'datasets/diamonds/diamonds_transformed.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    transformed_csv = luigi.Parameter(default=default_paths['transformed_csv'])


    def requires(self):
        # diamonds_cleaned.csv is needed
        return DataPreprocessing(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # From worst to best categories
        cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

        # Read diamonds_cleaned.csv
        df = pd.read_csv(self.input().path)

        logger.info('Retrieved the cleaned dataset')

        # Ordinal encoding on the categorical features
        df['cut'] = OrdinalEncoder(categories=[cut_order], dtype=int).fit_transform(df[['cut']])
        df['color'] = OrdinalEncoder(categories=[color_order], dtype=int).fit_transform(df[['color']])
        df['clarity'] = OrdinalEncoder(categories=[clarity_order], dtype=int).fit_transform(df[['clarity']])

        logger.info('Applied ordinal encoding to the categorical features')
        
        # Save to diamonds_transformed.csv
        df.to_csv(self.output().path, index=False)

        logger.info('Saved to csv the transformed dataset')
        logger.info(f'Finished task {self.__class__.__name__}')
    

    def output(self):
        return luigi.LocalTarget(self.transformed_csv)
    


class SplitDataset(BetterCompleteCheck, luigi.Task):
    """
    Splits the transformed dataset into training and testing sets.
    Outputs two files, one for training and one for testing.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw diamonds data. Default: 'datasets/diamonds/diamonds.csv'
    - train-csv: Path for the output training set (after preprocessing and transformation). Default: 'datasets/diamonds/diamonds_train.csv'
    - test-csv: Path for the output testing set (already cleaned too). Default: 'datasets/diamonds/diamonds_test.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    test_csv = luigi.Parameter(default=default_paths['test_csv'])
    

    def requires(self):
        # diamonds_transformed.csv is needed
        return DataTransformation(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Connect to the h2o cluster managed by the wrapper script
        h2o.connect()

        logger.info('Connected to h2o cluster')

        # Retrieve transformed dataframe
        h2o_df = h2o.import_file(self.input().path)

        logger.info('Retrieved the transformed dataset')

        # Split into train and test
        train, test = h2o_df.split_frame(ratios = [.8], seed=1234)

        logger.info('Split into training set and test set')

        # Save to diamonds_train.csv and diamonds_test.csv
        h2o.export_file(train, self.train_csv, force=True)
        h2o.export_file(test, self.test_csv, force=True)

        logger.info('Saved to csv the training set and test set')
        logger.info(f'Finished task {self.__class__.__name__}')
        

    def output(self):
        return {'train_csv': luigi.LocalTarget(self.train_csv),
                'test_csv': luigi.LocalTarget(self.test_csv)}



class EvalModel(BetterCompleteCheck, luigi.Task):
    """
    Trains up to 10 GBM models only on the training set and selects the best one, based on performance.
    Outputs the trained model file, which is not for production use.

    Parameters:
    - input-csv: Path for the raw dataset to model. Default: 'datasets/diamonds/diamonds.csv'
    - model-file: Path to save the trained model used for evaluation purposes. Default: 'models/eval_model'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    model_file = luigi.Parameter(default=default_paths['model_file'])


    def requires(self):
        # diamonds_train.csv and diamonds_test.csv are needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Connect to the h2o cluster managed by the wrapper script
        h2o.connect()

        logger.info('Connected to h2o cluster')

        # Retrieve the train set with proper column types for categorical features
        train = h2o.import_file(self.input()['train_csv'].path, 
                                col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})
        
        logger.info('Retrieved the training set')

        # Target
        y = 'price'

        # Features
        X = train.columns
        X.remove(y)

        # Train H2O AutoML with the training set to get the model used for evaluation
        aml = H2OAutoML(max_models=10, seed=1234, include_algos=['GBM'])
        aml.train(x=X, y=y, training_frame=train)

        logger.info('Trained up to 10 GBM models using H2O AutoML')

        # Save the best model as eval_model
        h2o.save_model(model=aml.leader, filename=self.output().path, force=True)

        logger.info('Saved to file the best model used for evaluation')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.model_file)
    


class PerformanceEval(BetterCompleteCheck, luigi.Task):
    """
    Evaluates the trained model's performance on the test dataset.
    Outputs a csv file with performance metrics like MSE, RMSE, MAE, etc.
    Unlike the other tasks, DeployModel doesn't depend on this and so it has to be executed separately.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw diamonds data. Default: 'datasets/diamonds/diamonds.csv'
    - metrics-csv: Path to save the performance test metrics csv file. Default: 'evaluation/metrics.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    metrics_csv = luigi.Parameter(default=default_paths['metrics_csv'])


    def requires(self):
        # eval_model, diamonds_train.csv and diamonds_test.csv are needed
        return {'model_file': EvalModel(input_csv=self.input_csv),
                'splitted_dataset_csv': SplitDataset(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Connect to the h2o cluster managed by the wrapper script
        h2o.connect()

        logger.info('Connected to h2o cluster')

        # Retrieve the model
        model = h2o.load_model(self.input()['model_file'].path)

        logger.info('Retrieved the evaluation model')

        # Retrieve the test set with proper column types for categorical features
        test = h2o.import_file(self.input()['splitted_dataset_csv']['test_csv'].path, 
                                col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})
        
        logger.info('Retrieved the test set')

        # Get the price predictions
        predictions = model.predict(test).round()

        logger.info('Got the predictions')

        # Convert from H2OFrame column to Pandas Series for compatibility
        with warnings.catch_warnings():
            # Suppress the warning about installing additional libraries
            warnings.simplefilter("ignore")
            actual_prices = test['price'].as_data_frame()['price']
            predicted_prices = predictions.as_data_frame()['predict']

        # Calculate the metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmsle = np.sqrt(mean_squared_log_error(actual_prices, predicted_prices))
        mean_residual_deviance = mse
        r2 = r2_score(actual_prices, predicted_prices)

        logger.info('Calculated the metrics on the test set')

        # Model name, it contains the timestamp too
        model_name = model.params['model_id']['actual']['name']

        # Dataframe containing the record
        metrics_df = pd.DataFrame({
            'model_name': [model_name],
            'mse': [mse],
            'rmse': [rmse],
            'mae': [mae],
            'rmsle': [rmsle],
            'mean_residual_deviance': [mean_residual_deviance],
            'r2': [r2]
        })

        # Create the path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        logger.info('Prepared the path and data (model name + metrics) for appending to csv')

        # Append the data to metrics.csv
        with open(self.output().path, 'a') as f:
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info('Appended to csv the model name and metrics')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.metrics_csv)



class DeployModel(BetterCompleteCheck, luigi.Task):
    """
    Deploys the model by re-training it on the entire dataset with the best hyperparameters from EvalModel.
    Outputs the deployment model file, ready for production use.
    
    Parameters:
    - input-csv: Path for the raw dataset to deploy the model on. Default: 'datasets/diamonds/diamonds.csv'
    - deploy-model-file: Path to save the deployment model. Default: 'models/deploy_model'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    deploy_model_file = luigi.Parameter(default=default_paths['deploy_model_file'])


    def requires(self):
        # eval_model and diamonds_transformed.csv are needed
        return {'model_file': EvalModel(input_csv=self.input_csv),
                'transformed_csv': DataTransformation(input_csv=self.input_csv)}
                
    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Connect to the h2o cluster managed by the wrapper script
        h2o.connect()

        logger.info('Connected to h2o cluster')

        # Retrieve the model
        model = h2o.load_model(self.input()['model_file'].path)

        logger.info('Retrieved the evaluation model')

        # Retrieve transformed dataframe with proper column types for categorical features
        h2o_df = h2o.import_file(self.input()['transformed_csv'].path,
                                 col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})
        
        logger.info('Retrieved the transformed dataset')

        # Freeze the hyperparameters from the original model
        params = model.actual_params

        # Remove early stopping parameters if they exist
        params_to_remove = ['stopping_rounds', 'stopping_metric', 'stopping_tolerance']
        for param in params_to_remove:
            params.pop(param, None)

        logger.info('Prepared the hyperparameters for the deployment model')

        # Train the model on all the available data with the same hyperparameters
        deploy_model = H2OGradientBoostingEstimator(**params)
        deploy_model.train(y='price', training_frame=h2o_df)

        logger.info('Got the deployment model trained on the whole dataset')

        # Save the model as deploy_model
        h2o.save_model(model=deploy_model, filename=self.output().path, force=True)

        logger.info('Saved to file the deployment model')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.deploy_model_file)



class FullPipeline(luigi.WrapperTask):
    """
    A wrapper task to run the full pipeline with default parameters.
    It ensures that both DeployModel and PerformanceEval are executed.

    This is used by the wrapper script for executing every single task of the pipeline.
    """
    def requires(self):
        return [DeployModel(), PerformanceEval()]