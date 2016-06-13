import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "LaminarFlow",
    version = "1.1.0.0",
    author = "Chris Dusold",
    author_email = "LaminarFlow@ChrisDusold.com",
    description = ("A meta class to wrap and automate TensorFlow."),
    license = read("LICENSE"),
    keywords = "TensorFlow",
    #url = "http://pyspeedup.rtfd.org/",
    packages=['laminarflow'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent", #Hopefully.
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
)
