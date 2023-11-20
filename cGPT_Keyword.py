import openai
import os
import pandas as pd
from collections import Counter
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load data
with open('data/data.txt', 'r') as file:
    text_data = file.read()

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

        # Logic to decide whether to use your data or GPT
        if should_use_my_data(user_input):
            response = get_my_response(user_input)
        else:
            response = get_gpt_response(user_input)

        print("GPT: ", response)

def should_use_my_data(input_text):
    keywords = extract_keywords(input_text)
    return any(keyword in text_data for keyword in keywords)

def get_my_response(input_text):
    # Define how you generate a response from your data
    return "Response based on my data"

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    frequency = Counter(words)
    # Extracting top 5 frequent words as keywords
    return [word for word, freq in frequency.most_common(5)]

if __name__ == "__main__":
    main()