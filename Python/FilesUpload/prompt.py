# Import the `os` module for working with the operating system
import os

# Import the `OpenAI` class for interacting with the OpenAI API
from langchain_openai.llms import OpenAI

# Import the `openai_key` variable from the `constants` module
# (Assuming `constants.py` holds your OpenAI API key securely)
from constants import openai_key

# Import `PromptTemplate` from `langchain.prompts` (for creating structured prompts)
from langchain.prompts import PromptTemplate

# Import `LLMChain` from `langchain.chains` (for building LLM workflows)
from langchain.chains import LLMChain

# Set OpenAI API key as environment variable (security note: consider env variables)
os.environ["OPENAI_API_KEY"] = openai_key

# Define a function to generate a movie name based on movie type / year
def generate_movie_name(movie_type,year):
  """Generates a movie name using the OpenAI Large Language Model (LLM).

  This function creates an OpenAI LLM object, uses a prompt template to influence
  the output, and returns the generated response.

  Args:
      movie_type (str): The type of movie to search for (e.g., "Horror", "Comedy").

  Returns:
      str: The generated movie name (or error message if unsuccessful).
  """

  # Create an OpenAI LLM object with a temperature of 0.7 (controls creativity)
  llm = OpenAI(temperature=0.7)

  # Define a prompt template with a placeholder for 'movie_type'
  prompt_template_name = PromptTemplate(input_variables=['movie_type','year'],
                                         template="Top must watch {movie_type} movies came in {year}")

  # Create an LLMChain that combines the LLM object and the prompt template
  movie_name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

  # Call the LLMChain with the 'movie_type' argument to generate the response
  response = movie_name_chain({'movie_type': movie_type,'year': year})

  # Return the generated response (the movie name)
  return response

# Check if the script is run directly (not imported as a module)
if __name__ == "__main__":

  # Call the generate_movie_name function to get a movie name for 'Horror' genre
  movie_name = generate_movie_name('Comedy','1980')

  # Print the generated movie name in a formatted way
  print(f"Top Must-Watch Movie Name: {movie_name}")





