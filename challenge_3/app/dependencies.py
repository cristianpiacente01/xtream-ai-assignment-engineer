"""
Dependencies module for the FastAPI application.
It includes utilities for path handling and model loading.

Cristian Piacente
"""

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import os
from configparser import ConfigParser
from pathlib import Path


def handle_path(path: str, challenge_2: bool = False) -> str:
    """
    Converts a given path string to an absolute path. 

    If the path is relative, it is resolved either against the challenge_2 or challenge_3 directory,
    based on the challenge_2 flag.

    Parameters:
    - path (str): The model file path.
    - challenge_2 (bool): Flag indicating whether to resolve the path against challenge_2 directory.

    Returns:
    - str: The resolved absolute filesystem path.
    """

    # string to Path conversion
    path = Path(path)

    # Handle relative path if the path is not absolute
    if not path.is_absolute():
        # Current file path
        current_file = Path(__file__).resolve()

        # Get the correct base directory
        if challenge_2:
            # Consider challenge_2 directory path
            root_dir = current_file.parent.parent.parent
            base_dir = root_dir / 'challenge_2'
        else:
            # Consider challenge_3 directory path
            base_dir = current_file.parent.parent

        # Create the absolute path
        path = base_dir / path

    # Return the filesystem absolute path
    return os.fspath(path)



def load_model(model_path: str = None) -> H2OGradientBoostingEstimator:
    """
    Loads the model from the specified path. 

    If no path is provided, it defaults to the path specified in the luigi.cfg file (from challenge 2).

    Parameters:
    - model_path (str): Optional; The path to the model file.

    Returns:
    - H2OGradientBoostingEstimator: The loaded model.
    """

    if model_path is None:
        # No argument passed via command line, use luigi.cfg
        config = ConfigParser()
        config.read(Path(__file__).parent.parent.parent / 'challenge_2' / 'luigi.cfg')
        model_path = config.get('DeployModel', 'deploy_model_file')
        model_path = handle_path(model_path, challenge_2=True)
    else:
        # Use the --model-path argument from command line
        model_path = handle_path(model_path)

    # Load the model
    model = h2o.load_model(model_path)
    
    # Returned the loaded model
    return model