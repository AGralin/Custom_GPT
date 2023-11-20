
import openai
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load data if necessary
# csv_data = pd.read_csv('data.csv')
text_data = open('data/data.txt').read()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_response(prompt):
    try:
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=prompt,
          max_tokens=150
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return ""

def main():
    while True:
        user_input = input("You: ")
        # Add logic here to decide whether to use your data or GPT for the response
        response = get_gpt_response(user_input)
        print("GPT: ", response)

if __name__ == "__main__":
    main()

