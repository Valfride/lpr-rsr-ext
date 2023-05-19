import time
import sys
import logging

class EventlogHandler():
    def __init__(self, function):
        self.function = function
        
    def __call__(self, *args, **kwargs):
        try:
            result = self.function(*args, **kwargs)
            return result
        except (Exception) as e:
            sys.exit('Error in {}: {}'.format(self.function.__name__, e))
                
        
        
class ExecTime():
    def __init__(self, function):
        self.function = function
    
    def __call__(self, *args, **kwargs):
        initial_time = time.time()
        result = self.function(*args, **kwargs)
        end_time = time.time()
        m, s = divmod(end_time - initial_time, 60)
        h, m = divmod(m, 60)
        print('\nTotal execution time: {:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s))
        
        return result