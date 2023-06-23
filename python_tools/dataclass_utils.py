'''

to explore the use cases of the dataclass module

--- source 1 ----
link: https://www.dataquest.io/blog/how-to-use-python-data-classes/#:~:text=In%20Python%2C%20a%20data%20class,a%20program%20or%20a%20system.

What is purpose of dataclass?
1) To easily create a class meant for just storing data that 
already has a bunch of the basic functions created for you
- __init__, __eq__, __repr__

2) with other options for the decorator (like order = True)
other functions would be implemented
__lt__ (less than), __le__ (less or equal), __gt__ (greater than), and __ge__ (greater or equal)

3) Could set the object to frozen so that nothing can be changed 

4) Can do inheritance from one dataclass to the next

How to use?
Just import the class and use it as a decorator


What did it use to look like? We used to have to implement a class from scratch like
class Person():
    def __init__(self, name='Joe', age=30, height=1.85, email='joe@dataquest.io'):
        self.name = name
        self.age = age
        self.height = height
        self.email = email

    def __eq__(self, other):
        if isinstance(other, Person):
            return (self.name, self.age,
                    self.height, self.email) == (other.name, other.age,
                                                 other.height, other.email)
        return NotImplemented


'''
from dataclasses import dataclass

def examples():

    # --- basic examples
    from typing import Tuple

    @dataclass
    class Person():
        name: str
        age: int
        height: float
        email: str
        house_coordinates: Tuple

    print(Person('Joe', 25, 1.85, 'joe@dataquest.io', (40.748441, -73.985664)))

    from typing import List

    @dataclass
    class People():
        people: List[Person]

    #-- example: redefining a method that already implemented
    @dataclass
    class Person():
        name: str
        age: int
        height: float
        email: str

        def __repr__(self):
            return (f'''This is a {self.__class__.__name__} called {self.name}.''')

    person = Person('Joe', 25, 1.85, 'joe@dataquest.io')
    print(person)


    # --- making a custom comparison method ---
    from dataclasses import dataclass, field

    @dataclass(order=True)
    class Person():
        sort_index: int = field(init=False, repr=False)
        name: str
        age: int
        height: float
        email: str

        def __post_init__(self):
            self.sort_index = self.age

    joe = Person('Joe', 25, 1.85, 'joe@dataquest.io')
    mary = Person('Mary', 43, 1.67, 'mary@dataquest.io')

    print(joe > mary)



    # --- example of how to freeze all attributes --
    @dataclass(frozen=True)
    class Person():
        name: str
        age: int
        height: float
        email: str

    joe = Person('Joe', 25, 1.85, 'joe@dataquest.io')

    joe.age = 35
    print(joe)


    # --- example of inheritance ---
    """
    Note, if first class has default attributes, 
    all attributes of inheriting class must have default attributes
    or else like having arguments with no default after having those
    with default
    """
    @dataclass
    class Person():
        name: str = 'Joe'
        age: int = 30
        height: float = 1.85
        email: str = 'joe@dataquest.io'

    @dataclass(order=True)
    class Employee(Person):
        salary: = 10 int
        departament: = "security" str

    print(Employee('Joe', 25, 1.85, 'joe@dataquest.io', 100000, 'Marketing'))
