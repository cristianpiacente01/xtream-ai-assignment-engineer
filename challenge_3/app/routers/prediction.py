"""
This module defines the prediction endpoints for the FastAPI application. 

The dependency injection mechanism provided by FastAPI is used to inject the loaded model, 
allowing the separation of concerns and making the application more modular and maintainable.

Cristian Piacente
"""

import h2o
import warnings
from fastapi import APIRouter, Depends
from pydantic import BaseModel, model_validator, PositiveFloat, conlist
from typing import Optional

from app.model import get_model
from app.logging_config import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/v1/prediction")


# Dictionary used for validating the input and encoding categorical values, from worst to best
ordinal_map = {
    'cut': {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4},
    'color': {chr(i): i - ord('D') for i in range(ord('D'), ord('Z') + 1)}, # From 'D': 0 to 'Z': 22
    'clarity': {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
}

# List of all possible features, used to handle missing values
possible_features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']


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

    carat: Optional[PositiveFloat] = None
    cut: Optional[str] = None
    color: Optional[str] = None
    clarity: Optional[str] = None
    depth: Optional[PositiveFloat] = None
    table: Optional[PositiveFloat] = None
    x: Optional[PositiveFloat] = None
    y: Optional[PositiveFloat] = None
    z: Optional[PositiveFloat] = None

    @model_validator(mode='before')
    def validate_input(cls, data: dict):
        """
        Validates the input data for diamond features before it's processed by the model.

        This function ensures that the passed categorical features are within the valid categories.

        Raises:
        - ValueError: If a categorical value is not valid.
        """
        
        # Validate categorical features
        for categorical_feature in ordinal_map.keys():
            if categorical_feature in data:
                # The categorical feature was passed 
                if not data[categorical_feature] in ordinal_map[categorical_feature]:
                    # ... but the value is not a valid category

                    logger.error(f'Invalid category found for {categorical_feature}: {data[categorical_feature]}')

                    raise ValueError(f"{data[categorical_feature]} is not a valid {categorical_feature}")
            
        logger.info('Diamond features validated successfully!')

        # All checks passed
        return data


    def to_model_input(self):
        """
        Converts the input data to a format suitable for prediction with the model.

        The categorical features are encoded based on the ordinal mapping. 

        Returns:
        - A H2OFrame ready for prediction.
        """

        # Dictionary without the missing values, which will be handled later
        model_input = {**self.dict(exclude_none=True)}

        # Apply ordinal encoding on the passed categorical features
        for categorical_field in ordinal_map.keys():
            if categorical_field in model_input: 
                model_input[categorical_field] = ordinal_map[categorical_field][model_input[categorical_field]]

                logger.info(f'Applied ordinal encoding for feature {categorical_field}')
        
        # Convert to H2OFrame
        h2o_frame = h2o.H2OFrame(model_input)

        # Handle the missing values directly on the H2OFrame
        for feature in possible_features:
            if feature not in model_input:
                # If a feature wasn't passed, add a column with a missing value
                h2o_frame[feature] = None

                logger.info(f'Feature {feature} wasn\'t passed and will contain a missing value')

        # Cast the categorical columns
        for categorical_field in ordinal_map.keys():
            h2o_frame[categorical_field] = h2o_frame[categorical_field].asfactor()

        # Reorder the H2OFrame columns, since some could have been added at the end (missing values)
        h2o_frame = h2o_frame[:, possible_features]
        
        return h2o_frame



class MultipleDiamondFeatures(BaseModel):
    """
    Pydantic model for representing features of multiple diamonds for bulk price prediction.

    Attributes:
    - diamonds (List[DiamondFeatures]): A list containing at least one or more DiamondFeatures instances.
    """

    diamonds: conlist(DiamondFeatures, min_length=1)  


    def to_model_input(self):
        """
        Converts the list of DiamondFeatures instances into a single H2OFrame suitable for batch prediction with the model.

        It combines the processed features of all diamonds into a single H2OFrame.

        Returns:
        - H2OFrame: A combined H2OFrame containing the processed features of all diamonds, ready for batch prediction.
        """

        # Retrieve the H2OFrame for each diamond
        frames = [diamond.to_model_input() for diamond in self.diamonds]

        # There is always at least one diamond
        combined_frame = frames[0]

        # Use rbind to combine all individual H2OFrame instances vertically
        for frame in frames[1:]:
            combined_frame = combined_frame.rbind(frame)

        logger.info(f'The combined H2OFrame contains {combined_frame.nrows} rows')
        
        # Cast the categorical columns and return the combined H2OFrame
        for categorical_field in ordinal_map.keys():
            combined_frame[categorical_field] = combined_frame[categorical_field].asfactor()

        return combined_frame



@router.post("/")
async def predict_single(features: DiamondFeatures, model=Depends(get_model)):
    """
    Endpoint for making predictions with the loaded model.
    The model is injected as a dependency.
    
    Parameters:
    - features (DiamondFeatures): A Pydantic model representing the required features of a diamond.
    - model (Depends(get_model)): Dependency that injects the loaded model for prediction.

    Returns:
    - dict: A dictionary with a key predicted_price associated with the predicted price of the diamond.
    """

    # Get H2OFrame for compatibility with the predict function
    model_input = features.to_model_input()

    logger.info('Got H2OFrame from the diamond features')

    # Predict the diamond's price (float value)
    prediction = model.predict(model_input).getrow()[0]

    # Round the result to get an integer and return it
    prediction = round(prediction)

    logger.info(f'Got the predicted price: {prediction}')

    return {"predicted_price": prediction}



@router.post("/bulk")
async def predict_bulk(features: MultipleDiamondFeatures, model=Depends(get_model)):
    """
    Endpoint for making bulk predictions on multiple diamonds.

    This endpoint accepts features for multiple diamonds and uses the loaded model, injected as a dependency. 

    IMPORTANT: missing values are accepted, BUT every feature must contain at least a value among the diamonds.
    It means that a column can't have only missing values, it will lead to an error when invoking the predict function.

    Parameters:
    - features (MultipleDiamondFeatures): A Pydantic model representing the required features for multiple diamonds.
    - model (Depends(get_model)): Dependency that injects the loaded model for prediction.

    Returns:
    - dict: A dictionary with a key predicted_prices associated with a list of predicted prices for the diamonds.
    """

    # Get H2OFrame containing all the diamonds
    model_input = features.to_model_input()

    logger.info('Got H2OFrame with all the passed diamonds')

    # Perform prediction to get the prices
    predictions = model.predict(model_input)

    # Round the prices to get integers
    predictions = predictions.round()

    with warnings.catch_warnings():
            # Suppress the warning about the conversion, as usual
            warnings.simplefilter("ignore")

            # Convert to list, to return the values on the response body
            predictions = h2o.as_list(predictions)['predict'].tolist()

    logger.info(f'Got the predicted prices: {predictions}')

    return {"predicted_prices": predictions}