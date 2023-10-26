"""
Ex: Can create an abstract class with an
abstract function and define functionality in that
abstract function, and then reference that functionality
when overriding the abstract method in the child class
"""

from abc import (
  ABC,
  abstractmethod,)

class Dog(ABC):
    
    @abstractmethod
    def bark(self):
        print("hello")
        
class Doggy(Dog):
    def __init__(self):
        self.name = "Bob"
    def bark(self):
        super().bark()
        print('hi')

