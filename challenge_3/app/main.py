"""
Main module for the FastAPI application.

It exploits a pre-trained H2O AutoML model to predict diamond prices based on their features. 

The application initializes a h2o cluster, loads the model, and sets up API endpoints for making predictions.

It supports loading a model from a custom path passed via command line, 
or defaults to the one specified in the luigi.cfg configuration file (from challenge_2). 

Cristian Piacente
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import h2o
import sys

from app.dependencies import load_model
from app.routers import prediction
from app.model import set_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for the application's lifespan. 
    Initializes and shuts down the h2o cluster, and loads the model.

    Parameters:
    - app (FastAPI): The FastAPI application instance.
    """

    # Initialize h2o cluster
    h2o.init()

    # Load the model
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
    else:
        model = load_model()

    # Set the model to use it globally
    set_model(model)

    yield

    # Shut down h2o cluster
    if h2o.cluster() is not None:
        h2o.cluster().shutdown()



# FastAPI application instance, created with the defined lifespan
app = FastAPI(lifespan=lifespan)

# Bind the routers to the FastAPI application
app.include_router(prediction.router)



if __name__ == "__main__":
    uvicorn.run('app.main:app')