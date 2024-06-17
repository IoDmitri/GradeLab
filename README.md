The following is GradeLab, a package based on the paper: [insert paper link here]

## Intro
GradeLab is a package for evaluating the capabilities of LLMs when they're deciding on multiple choice options. The Score is based on the following formula: 

$$
\text{Grade Score} = \frac{2 \times (\text{llm Score} \times \text{Choice Score})}{\text{llm Score} + \text{Choice Score}}
$$


A high score indicates that an LLM exhibits low order bias and high consistency with selecting the same output over and over again, even in the face of re-arranging the order of options

## Usage Instructions
The following is the options for running an experiment: 

```
python experiment.py --help
usage: experiment.py [-h] [--is_local] [--random_option] [--api_key API_KEY] [--temperature TEMPERATURE] [--model MODEL] [--save_path SAVE_PATH]
                     [--url URL]
                     dataset prompt client prompt_key outputs_key

Evaluate dataset with LLMs.

positional arguments:
  dataset               Name of the dataset or path if local.
  prompt                Prompt type or custom prompt.
  client                Client to use: 'openai', 'mistral', 'anthropic' or 'together'
  prompt_key            Key for the prompt in the dataset.
  outputs_key           Key for the outputs in the dataset.

options:
  -h, --help            show this help message and exit
  --is_local            Flag to load dataset from disk.
  --random_option       Flag for using a random option.
  --api_key API_KEY     API key for the client.
  --temperature TEMPERATURE
                        Temperature for generation.
  --model MODEL         Model to be used for generation.
  --save_path SAVE_PATH
                        Path to save the output.
  --url URL             the url to use for an OpenAI compatible API
dmitri@Dmitris-MacBook-Pro-2 GradeLab % 
```

Example Usage
```
python experiment.py IoDmitri/OAST-full \  
prompts\judge_prompt_3.txt\  
 openai \   
 prompt \   
 outputs \    
 --api_key 1234  \  
 --save_path gpt_4_prompt3.json\   
 --random_option
```
