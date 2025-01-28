#!/usr/bin/env python
import sys
import warnings

from crew import RagDevelopment

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    user_input = input("Enter your question: ")
    inputs = {
        'input': user_input
    }
    result = RagDevelopment().crew().kickoff(inputs=inputs)
    print(result)
    #RagDevelopment().crew().kickoff()

run()