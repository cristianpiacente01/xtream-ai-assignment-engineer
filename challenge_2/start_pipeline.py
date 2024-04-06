"""
Pipeline wrapper script for managing h2o cluster efficiently.

Cristian Piacente
""" 

import subprocess
import h2o
import logging


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('luigi-wrapper')


def main():
    try:
        # Initialize h2o cluster
        h2o.init()
        
        logger.info('Initialized h2o cluster')

        # Execute Luigi pipeline with default parameters
        subprocess.run(["python", "-m", "luigi", "--module", "pipeline", "FullPipeline", "--local-scheduler"], check=True)

        logger.info('Run the pipeline successfully!')
    
    except Exception as e:
        logger.error(f'An error occurred: {e}')

    finally:
        # Always shut down h2o cluster
        if h2o.cluster() is not None:
            h2o.cluster().shutdown()

            logger.info('Shut down h2o cluster')



if __name__ == "__main__":
    main()