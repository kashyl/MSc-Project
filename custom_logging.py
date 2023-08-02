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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'logs')  # Directory where the log file will be stored
LOG_PATH = os.path.join(LOG_DIR, 'setup.log')


log = None


def log_function_call(func):
    """
    Decorator for Logging Function Calls.
    A decorator can log the start and end of a function call, including the elapsed time.
    It also supports async functions.
    """

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        log.info(f"Starting {func.__name__} ...")

        result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = "{:.1f}".format(end_time - start_time)
        log.info(f"Finished {func.__name__}. Total time: {elapsed_time} seconds")
        return result

    return wrapper

def setup_logging(clean=False, debug=False):
    global log
    
    if log is not None:
        return log
    
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
    
    log = logging.getLogger("sd")
    log.addHandler(rh)
    
    return log
