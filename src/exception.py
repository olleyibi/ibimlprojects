import sys

# -------------------------------------------------------------
# Function: error_message_detail
# Purpose:  Generates a detailed error message that includes:
#           - The Python file name where the error occurred
#           - The line number of the error
#           - The original error message
# -------------------------------------------------------------
def error_message_detail(error, error_detail: sys):
    # Extract traceback details (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name where the exception was raised
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format a detailed error message with script name, line, and error message
    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] "
        "error message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))
    
    return error_message


# -------------------------------------------------------------
# Class: CustomException
# Purpose: A custom exception class that extends Python's
#          built-in Exception. It captures and formats
#          error details for improved debugging and logging.
# -------------------------------------------------------------
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception class
        super().__init__(error_message)
        
        # Generate a formatted error message using the helper function
        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail
        )

    def __str__(self):
        # Return the formatted error message when printed or logged
        return self.error_message
