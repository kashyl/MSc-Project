import time
from threading import Event

class TimerApp:
    def __init__(self):
        self._start_time = None
        self._is_running = False
        self._stop_event = Event()

    def _reset(self):
        """Reset the state of the timer."""
        self._is_running = False
        self._start_time = None
        
    def start_timer(self):
        if not self._is_running:
            # Prepare for timer start
            self._stop_event.clear()
            self._is_running = True
            self._start_time = time.time()
            
            # Timer loop
            for second in range(1, 6):
                if self._stop_event.is_set():
                    elapsed_time = time.time() - self._start_time
                    # yield f"Timer stopped at {elapsed_time:.2f} seconds"
                    self._reset()
                    return
                
                time.sleep(1)
                yield second
            
            # Check if timer was not stopped before completion
            if not self._stop_event.is_set():
                # yield "boom"
                yield 0
            self._reset()
        
    def stop_timer(self):
        """Stop the timer if it's running."""
        if self._is_running:
            self._stop_event.set()
            elapsed_time = time.time() - self._start_time
            return f"Timer manually stopped at {elapsed_time:.2f} seconds"
