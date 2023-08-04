import os
import logging
from filelock import FileLock
import time
import asyncio

from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

from functools import wraps

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')  # Directory where the log file will be stored
LOG_PATH = os.path.join(LOG_DIR, 'setup.log')


logger = None

def log_function_call(func):
    """
    Decorator to log information about function calls.

    This decorator logs the start and end of a function call, including the elapsed time, for both
    synchronous and asynchronous functions. It prints log messages indicating the start of the function,
    its name, and the total time taken to execute the function in seconds.

    Parameters:
        func (callable or coroutine function): The function or coroutine function to be wrapped.

    Returns:
        callable: A wrapper function that logs information about the call.

    Usage:
        @log_function_call
        def sync_function():
            pass

        @log_function_call
        async def async_function():
            pass
    """
    def log_info(func_name, start_time):
        end_time = time.time()
        elapsed_time = "{:.1f}".format(end_time - start_time)
        logger.info(f"Finished {func_name}. Total time: {elapsed_time} seconds")

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__} ...")
            result = await func(*args, **kwargs)
            log_info(func.__name__, start_time)
            return result

        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__} ...")
            result = func(*args, **kwargs)
            log_info(func.__name__, start_time)
            return result

        return sync_wrapper

def setup_logging(clean=False, debug=False):
    global logger
    
    if logger is not None:
        return logger
    
    try:
        if clean and os.path.isfile(LOG_PATH):
            with FileLock(LOG_PATH):
                if os.path.isfile(LOG_PATH): # Check again after acquiring the lock
                    os.remove(LOG_PATH)
    except:
        pass

     # Make sure the directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s', 
        filename=LOG_PATH, 
        filemode='a', 
        encoding='utf-8', 
        force=True
    )
    
    console = Console(
        log_time=True, 
        log_time_format='%H:%M:%S-%f', 
        theme=Theme({
            "traceback.border": "black",
            "traceback.border.syntax_error": "black",
            "inspect.value.border": "black",
        })
    )

    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[])

    rh = RichHandler(
        show_time=True, 
        omit_repeated_times=False, 
        show_level=True, 
        show_path=False, 
        markup=False, 
        rich_tracebacks=True, 
        log_time_format='%H:%M:%S-%f', 
        level=logging.DEBUG if debug else logging.INFO, 
        console=console
    )

    rh.set_name(logging.DEBUG if debug else logging.INFO)
    
    logger = logging.getLogger("sd")
    logger.addHandler(rh)
    
    return logger

logger = setup_logging(debug=True)
