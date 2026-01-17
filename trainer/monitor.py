from typing import List, Dict, Any,Literal


class Monitor:
    def __init__(self,target: str,divisor: int = 1):
        self.value = None
        self.target = target
        self.divisor = divisor
    
    def exists(self,data):
        return self.target in data

    def update(self,data,mode: Literal["max","min","always","unequal"]):
        if not self.exists(data):
            return False

        new_value = data[self.target]

        if mode == "always":
            self.value = new_value
            return True
        elif mode == "unequal":
            if self.value is None or new_value != self.value:
                self.value = new_value
                return True
        elif mode == "max":
            if self.value is None or new_value > self.value:
                self.value = new_value
                return True
        elif mode == "min":
            if self.value is None or new_value < self.value:
                self.value = new_value
                return True
    
    def divisible(self,data):
        if not self.exists(data):
            return False
        
        new_value = data[self.target]
        if new_value % self.divisor == 0:
            return True
        else:
            return False