import sys
from src.logger import logging

def error_message_info(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_info(error_message, error_detail)

    def __str__(self):
        return self.error_message

    def log_error(self):
        logging.error(self.error_message)  # Log the full error message

# Main block should use double underscores in "__main__"
if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise ZeroDivisionError
    except Exception as e:
        custom_exception = CustomException(e, sys)
        custom_exception.log_error()  # Log the error using the new method
        raise custom_exception  # Raise the custom exception again
