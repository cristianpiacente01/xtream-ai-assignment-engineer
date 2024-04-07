"""
Module that provides functionality for managing the global state of the model used for predictions. 

It ensures that a h2o Gradient Boosting Machine model can be accessed across different modules of the FastAPI application, 
avoiding the risk of circular dependencies.

Cristian Piacente
"""

from typing import Optional
from h2o.estimators import H2OGradientBoostingEstimator


# Global variable to hold the loaded h2o model (None if no model is loaded)
model: Optional[H2OGradientBoostingEstimator] = None


def set_model(new_model: H2OGradientBoostingEstimator) -> None:
    """
    Sets the global model variable to a new h2o model.
    
    Parameters:
    - new_model (H2OGradientBoostingEstimator): The new model to set as the global model.
    """

    global model
    model = new_model



def get_model() -> Optional[H2OGradientBoostingEstimator]:
    """
    Retrieves the currently loaded h2o model.
    
    Returns:
    - Optional[H2OGradientBoostingEstimator]: The currently loaded h2o model, or None if no model is loaded.
    """

    return model