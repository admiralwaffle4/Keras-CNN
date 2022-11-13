import json
import pandas as pd


class Person:
    name = ""
    age = 18

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old."


def factorial(n: int):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


if __name__ == '__main__':
    f = pd.read_json("people.json")
