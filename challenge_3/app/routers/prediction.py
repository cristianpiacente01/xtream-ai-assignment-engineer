"""
This module defines the prediction endpoints for the FastAPI application. 

The dependency injection mechanism provided by FastAPI is used to inject the loaded model, 
allowing the separation of concerns and making the application more modular and maintainable.

Cristian Piacente
"""

import h2o
from fastapi import APIRouter, Depends
from pydantic import BaseModel, model_validator
from typing import Optional

from app.model import get_model


router = APIRouter(prefix="/v1/prediction")


# Dictionary used for validating the input and encoding categorical values, from worst to best
ordinal_map = {
    'cut': {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4},
    'color': {chr(i): i - ord('D') for i in range(ord('D'), ord('Z') + 1)}, # From 'D': 0 to 'Z': 22
    'clarity': {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
}


class DiamondFeatures(BaseModel):
    """
    Pydantic model representing the features of a diamond required for price prediction.

    Each attribute represents a characteristic of the diamond, with categorical values
    for cut, color, and clarity, and numerical values for carat, depth, table, x, y, z.

    Fields can be omitted in requests.

    Here is a detailed list of the possible features:
    - carat - weight of the diamond 
    - cut - quality of the cut, it can be Fair (worst), Good, Very Good, Premium, Ideal (best)
    - color - diamond color, it can be D (worst), E, F, G, H, I, J, ..., Z (best)
    - clarity - how clear the diamond is, it can be I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)
    - depth - depth percentage of the diamond
    - table - table percentage of the diamond
    - x - diamond length in mm
    - y - diamond width in mm
    - z - diamond depth in mm
    """

    carat: Optional[float] = None
    cut: Optional[str] = None
    color: Optional[str] = None
    clarity: Optional[str] = None
    depth: Optional[float] = None
    table: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    @model_validator(mode='before')
    def validate_input(cls, data: dict):
        """
        Validates the input data for diamond features before it's processed by the model.

        This function ensures that the passed categorical features are within the valid categories,
        and also that the passed numerical features are strictly positive. 

        Raises:
        - ValueError: If a categorical value is not valid or a numberical value is negative or zero.
        """
        
        # Validate categorical features
        for categorical_feature in ordinal_map.keys():
            if categorical_feature in data:
                # The categorical feature was passed 
                if not data[categorical_feature] in ordinal_map[categorical_feature]:
                    # ... but the value is not a valid category
                    raise ValueError(f"{data[categorical_feature]} is not a valid {categorical_feature}")
    
        # Validate numerical features
        for numerical_feature in ['carat', 'depth', 'table', 'x', 'y', 'z']:
            if numerical_feature in data and data[numerical_feature] <= 0:
                # The numerical feature was passed but negative or zero
                raise ValueError(f"{numerical_feature} must be greater than 0")
            
        # All checks passed
        return data


    def to_model_input(self):
        """
        Converts the input data to a format suitable for prediction with the model.

        The categorical features are encoded based on the ordinal mapping. 

        Returns:
        - A H2OFrame ready for prediction.
        """

        # Dict without the None values
        model_input = {**self.dict(exclude_none=True)}

        # Apply ordinal encoding on the passed categorical features
        for categorical_field in ['cut', 'color', 'clarity']:
            if categorical_field in model_input: 
                model_input[categorical_field] = ordinal_map[categorical_field][model_input[categorical_field]]
        
        # From dict to H2OFrame
        h2o_frame = h2o.H2OFrame(model_input)

        # Cast the passed categorical columns and return the H2OFrame
        for categorical_field in ['cut', 'color', 'clarity']:
            if categorical_field in model_input: 
                h2o_frame[categorical_field] = h2o_frame[categorical_field].asfactor()
        
        return h2o_frame



@router.post("/")
async def predict(features: DiamondFeatures, model=Depends(get_model)):
    """
    Endpoint for making predictions with the loaded H2O model.
    The model is injected as a dependency.
    
    Parameters:
    - features (DiamondFeatures): A Pydantic model representing the required features of a diamond.
    - model (Depends(get_model)): Dependency that injects the loaded H2O model for prediction.

    Returns:
    - dict: A dictionary with a key 'predicted_price' associated with the predicted price of the diamond.
    """

    # Get H2OFrame for compatibility with the predict function
    model_input = features.to_model_input()

    # Predict the diamond's price (float value)
    prediction = model.predict(model_input).getrow()[0]

    # Round the result to get an integer and return it
    prediction = round(prediction)

    return {"predicted_price": prediction}