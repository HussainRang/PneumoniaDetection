from config import LOGS_PATH
import logging
import os

def setup_logs():

    try:

        if("logs" not in os.listdir(".")):
            os.mkdir(LOGS_PATH)
        
        if( len( os.listdir(LOGS_PATH) ) != 0 ):
            with open(f"{LOGS_PATH}/logs.log","w") as logs_file:
                logs_file.truncate()
 
        return
    
    except Exception as e:
        print(e)
        