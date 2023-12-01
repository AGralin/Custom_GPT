# chatgpt-retrieval

Simple script to use ChatGPT on your own files.

[YouTube Video](https://youtu.be/9AXP7tCI9PI).

## Usage Instructions (Local):

### 1. Open AlgoTrading in a code editor such as VS Code and add a .env file in the GPT_ENV folder. Add your API key to the folder in the format below.
* make sure to add the .env to a .gitignore file if you plan on pushing to github
```
OPENAI_API_KEY = 'YOUR-SECRET-KEY-HERE'
```
### 2. Open CLI and cd into AlgoTrading/GPT_ENV
```
cd AlgoTrading/GPT_ENV
```

### 3. Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.
```
pip install langchain openai chromadb tiktoken unstructured
```

### 3. Test reading custom data
```
python3 CGPT.py "What is my dog's name?"

Response: Your dog's name is Rex.
```
### 4. Test OpenAI API
```
python3 CGPT.py "Who was the first President of the USA?"

Response: The first president of the USA was George Washington.
```
### 5. Add data
Place your data into `data/data.txt`. (.txt or .csv files only)