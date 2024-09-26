import sys
import logging

# Configure logging
logging.basicConfig(filename='C:\\ML Projects\\error.log', level=logging.ERROR)

def error_message_info(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()               
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}] ".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_info(error_message, error_detail=error_detail)


    def __str__(self):
        return self.error_message


if __name__=="main":
    
    try:
        a=1/0
    except:
        logging.info("Zero Division Error")
        raise CustomException