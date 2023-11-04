from pipelines.ml_pipeline import ml_pipeline
from utils.setup_logs import setup_logs

import os
import logging 
import logging.config


# Setting up for Logging 
setup_logs()
logging.basicConfig( 
    level=logging .DEBUG , 
    filename='./logs/logs.log' , 
    format='%(asctime)s,%(msecs)d %(levelname)-8s %(filename)s [%(pathname)s:%(lineno)d in ' \
           'function %(funcName)s] \n %(message)s \n\n',
    datefmt='%Y-%m-%d:%H:%M:%S', )
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


if __name__=="__main__":
    logging.debug("Starting the pipeline")
    ml_pipeline()
        