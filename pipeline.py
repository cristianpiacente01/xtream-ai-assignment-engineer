# TODO try to optimize the code, then doc and README
# (find out if h2o cluster can only be started once in the pipeline, etc.)

# Cristian Piacente

import luigi # Since xtream uses it, as written on https://xtreamers.io/portfolio/power-forecasting/

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator

import os



class BetterCompleteCheck:
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
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    cleaned_csv = luigi.Parameter(default='datasets/diamonds/diamonds_cleaned.csv')
    

    def requires(self):
        # diamonds.csv is needed, use a fake task
        class FakeTask(luigi.Task):
            def output(_):
                return luigi.LocalTarget(self.input_csv)
        
        return FakeTask()
    

    def run(self):
        # Read diamonds.csv
        df = pd.read_csv(self.input_csv)

        # Drop rows with missing values (even though there aren't any in the original dataset)
        df.dropna(inplace=True)

        # Drop duplicated rows
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)

        # Drop inconsistent rows
        inconsistent_rows = df[(df['price'] <= 0) | (df['x'] == 0) | (df['y'] == 0) | (df['z'] == 0)]
        df = df[~df.index.isin(inconsistent_rows.index)]

        # Save to diamonds_cleaned.csv
        df.to_csv(self.output().path, index=False)


    def output(self):
        return luigi.LocalTarget(self.cleaned_csv)
    


class DataTransformation(BetterCompleteCheck, luigi.Task):
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    transformed_csv = luigi.Parameter(default='datasets/diamonds/diamonds_transformed.csv')


    def requires(self):
        # diamonds_cleaned.csv is needed
        return DataPreprocessing(input_csv=self.input_csv)
    

    def run(self):
        # From worst to best categories
        cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

        # Read diamonds_cleaned.csv
        df = pd.read_csv(self.input().path)

        # Ordinal encoding on the categorical features
        df['cut'] = OrdinalEncoder(categories=[cut_order]).fit_transform(df[['cut']])
        df['color'] = OrdinalEncoder(categories=[color_order]).fit_transform(df[['color']])
        df['clarity'] = OrdinalEncoder(categories=[clarity_order]).fit_transform(df[['clarity']])

        # A casting here would be useless: it's necessary to handle the types later
        
        # Save to diamonds_transformed.csv
        df.to_csv(self.output().path, index=False)
    

    def output(self):
        return luigi.LocalTarget(self.transformed_csv)
    


class SplitDataset(BetterCompleteCheck, luigi.Task):
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    train_csv = luigi.Parameter(default='datasets/diamonds/diamonds_train.csv')
    test_csv = luigi.Parameter(default='datasets/diamonds/diamonds_test.csv')
    

    def requires(self):
        # diamonds_transformed.csv is needed
        return DataTransformation(input_csv=self.input_csv)
    

    def run(self):
        # Initialize h2o cluster
        h2o.init()

        # Retrieve transformed dataframe
        h2o_df = h2o.import_file(self.input().path)

        # Split into train and test
        train, test = h2o_df.split_frame(ratios = [.8], seed=1234)

        # Save to diamonds_train.csv and diamonds_test.csv
        h2o.export_file(train, self.train_csv, force=True)
        h2o.export_file(test, self.test_csv, force=True)

        # Shut down h2o cluster
        h2o.cluster().shutdown()
        

    def output(self):
        return {'train_csv': luigi.LocalTarget(self.train_csv),
                'test_csv': luigi.LocalTarget(self.test_csv)}



class EvalModel(BetterCompleteCheck, luigi.Task):
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    model_file = luigi.Parameter(default='models/eval_model')


    def requires(self):
        # diamonds_train.csv and diamonds_test.csv are needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        # Initialize h2o cluster
        h2o.init()

        # Retrieve the train set with proper column types for categorical features
        train = h2o.import_file(self.input()['train_csv'].path, 
                                col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})

        # Target
        y = 'price'

        # Features
        X = train.columns
        X.remove(y)

        # Train H2O AutoML with the training set to get the model used for evaluation
        aml = H2OAutoML(max_models=10, seed=1234, include_algos=['GBM'])
        aml.train(x=X, y=y, training_frame=train)

        # Save the best model as eval_model
        h2o.save_model(model=aml.leader, filename=self.output().path, force=True)

        # Shut down h2o cluster
        h2o.cluster().shutdown()


    def output(self):
        return luigi.LocalTarget(self.model_file)
    


class PerformanceEval(BetterCompleteCheck, luigi.Task):
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    metrics_csv = luigi.Parameter(default='evaluation/metrics.csv')


    def requires(self):
        # eval_model, diamonds_train.csv and diamonds_test.csv are needed
        return {'model_file': EvalModel(input_csv=self.input_csv),
                'splitted_dataset_csv': SplitDataset(input_csv=self.input_csv)}


    def run(self):
        # Initialize h2o cluster
        h2o.init()

        # Retrieve the model
        model = h2o.load_model(self.input()['model_file'].path)

        # Retrieve the test set with proper column types for categorical features
        test = h2o.import_file(self.input()['splitted_dataset_csv']['test_csv'].path, 
                                col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})

        # Get the price predictions
        predictions = model.predict(test).round()

        # Convert from H2OFrame column to Pandas Series for compatibility
        actual_prices = test['price'].as_data_frame()['price']
        predicted_prices = predictions.as_data_frame()['predict']

        # Calculate the metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmsle = np.sqrt(mean_squared_log_error(actual_prices, predicted_prices))
        mean_residual_deviance = mse
        r2 = r2_score(actual_prices, predicted_prices)

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

        # Append the data to metrics.csv
        with open(self.output().path, 'a+') as f:
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a+', header=f.tell()==0, index=False, lineterminator='\n')

        # Shut down h2o cluster
        h2o.cluster().shutdown()


    def output(self):
        return luigi.LocalTarget(self.metrics_csv)



class DeployModel(BetterCompleteCheck, luigi.Task):
    input_csv = luigi.Parameter(default='datasets/diamonds/diamonds.csv')
    deploy_model_file = luigi.Parameter(default='models/deploy_model')


    def requires(self):
        # eval_model and diamonds_transformed.csv are needed
        return {'model_file': EvalModel(input_csv=self.input_csv),
                'transformed_csv': DataTransformation(input_csv=self.input_csv)}
                
    
    def run(self):
        # Initialize h2o cluster
        h2o.init()

        # Retrieve the model
        model = h2o.load_model(self.input()['model_file'].path)

        # Retrieve transformed dataframe with proper column types for categorical features
        h2o_df = h2o.import_file(self.input()['transformed_csv'].path,
                                 col_types = {'cut': 'enum', 'color': 'enum', 'clarity': 'enum'})

        # Freeze the hyperparameters from the original model
        params = model.actual_params

        # Train the model on all the available data with the same hyperparameters
        deploy_model = H2OGradientBoostingEstimator(**params)
        deploy_model.train(y='price', training_frame=h2o_df)

        # Save the model as deploy_model
        h2o.save_model(model=deploy_model, filename=self.output().path, force=True)

        # Shut down h2o cluster
        h2o.cluster().shutdown()


    def output(self):
        return luigi.LocalTarget(self.deploy_model_file)