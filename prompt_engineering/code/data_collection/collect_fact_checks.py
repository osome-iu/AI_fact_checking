"""
Purpose: 
    Collect ChatGPT 3.5 fact checks given multiple prompts.

    Run via: python collect_fact_checks.py

Inputs:
    None

Outputs:
    Three new files for the fact checking results of each of the tested prompt styles.
    Format: fact_checks_{PROMPT_TYPE}.jsonl
        See PROMPTS below for the different prompts.

Author: Matthew DeVerna
"""

import json
import os

import pandas as pd

from tenacity import retry, stop_after_attempt, wait_random_exponential
from json import JSONDecodeError

# Set up OpenAI client
from openai import OpenAI

api_key = os.environ["OPENAI_OSOME_API_KEY"]
openai_client = OpenAI(api_key=api_key)

# Make sure we are in the proper directory for the relative output dirs/files
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != CURR_DIR:
    os.chdir(CURR_DIR)

HEADLINES_FILE = "../../../data/stimuli/all_experimental_stimuli_meta_data.csv"
OUTPUT_DIR = "../../data/intermediate_files/"

JSON_BINARY_INSTRUCTIONS = (
    "You are a helpful AI fact checker. "
    "You will be provided with a news headline and your task is to fact check whether the headline is 'True' or 'False'. "
    "Make sure to provide your response as a JSON object that contains only one key called 'judgment'. "
    "Values for 'judgment' can only be 'True' or 'False'. "
    "If you are unsure, you should make your best guess between these two options."
)

JSON_RATIONALE_INSTRUCTIONS = (
    "You are a helpful AI fact checker. "
    "You will be provided with a news headline and your task is to fact check whether the headline is 'True' or 'False' and explain your rationale. "
    "Make sure to provide your response in a JSON object that contains two keys: 'judgment' and 'rationale'. "
    "Values for 'judgment' can only be 'True' or 'False'. "
    "If you are unsure, you should make your best guess between these two options. "
    "Values for 'rationale' should be concise and contain only pertinent information."
)

PROMPTS = {
    "prompt_original": {
        "system_content": None,
        "user_content": "I saw something today that claimed '{}'. Do you think that this is likely to be true?",
    },
    "prompt_json_binary": {
        "system_content": JSON_BINARY_INSTRUCTIONS,
        "user_content": "Article headline: {}",
    },
    "prompt_json_rationale": {
        "system_content": JSON_RATIONALE_INSTRUCTIONS,
        "user_content": "Article headline: {}",
    },
}


# Decorator applies exponential backoffs to the function, retrying up to six times
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_no_instructions(prompt, model="gpt-3.5-turbo", raw=True):
    """
    Make an OpenAI chat completions call with no system instructions.

    Parameters:
    ------------
    - prompt (str) : The prompt to pass to the model.
    - model (str): An OpenAI chat completions model (default = gpt-3.5-turbo).
    - raw (bool): True (default): returns the OpenAI ChatCompletion object directly.
        False: return the response text only.

    Returns
    ------------
    - If raw == True (default): returns the OpenAI ChatCompletion object directly.
    - If raw == False: return the response text only.
    """
    try:
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        response = openai_client.chat.completions.create(
            model=model, messages=messages, temperature=0
        )
        if raw:
            return response
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Error: {e}")
        raise


# Decorator applies exponential backoffs to the function, retrying up to six times
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_json_w_instructions(
    system_content, user_content, model="gpt-3.5-turbo", raw=True
):
    """
    Make an OpenAI chat completions call with system instructions and user content.
    Requests JSON object response format for more control.

    Parameters:
    ------------
    - system_content (str) : The system instructions.
    - user_content (str) : The user content.
    - model (str): An OpenAI chat completions model (default = gpt-3.5-turbo).
    - raw (bool): True (default): returns the OpenAI ChatCompletion object directly.
        False: return the response text only.

    Returns
    ------------
    - If raw == True (default): returns the OpenAI ChatCompletion object directly.
    - If raw == False: return the response text only.
    """
    try:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        response = openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,
        )
        if raw:
            return response
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Error: {e}")
        raise


def get_nested_attr(obj, attr_path, default=None):
    """
    Retrieve a nested attribute from an object. Works like get_dict_val but for
    classes with nested attributes.

    Parameters:
    ------------
    - obj: The object from which to get the nested attribute.
    - attr_path (list): The nested attribute path, as a list e.g., ["attr1", "attr2"].
    - default: The default value to return if any attribute in the path doesn't exist.

    Returns:
    -----------
    - The value of the nested attribute if it exists, otherwise 'default'.
    """
    current = obj
    for attr in attr_path:
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return default
    return current


def parse_response(response):
    """
    Parses the OpenAI ChatCompletions response object to extract
    specific keys into a simplified dictionary.

    Parameters:
    ------------
    - response: A OpenAI ChatCompletions response object from OpenAI gpt-3.5-turbo

    Returns:
    -----------
    - clean_response (dict): a parsed version of response
        - keys are:
            - response_text (str) : text response from OpenAI model
            - finish_reason (str) : why the model stopped. ("stop" is what we want)
            - total_tokens (int): total number of tokens processed in query
            - completion_tokens (int): number of tokens utilized for completiong
            - prompt_tokens (int): number of tokens utilized in prompt
            - time_created (int): unix timestamp of when the query was called
            - model (str): name of the model utilized
    """
    # Initialize the simplified dictionary
    clean_response = {"response_text": None, "finish_reason": None}

    # Extract the message response
    if hasattr(response, "choices") and len(response.choices) > 0:
        clean_response["response_text"] = get_nested_attr(
            response.choices[0], ["message", "content"], None
        )
        clean_response["finish_reason"] = get_nested_attr(
            response.choices[0], ["finish_reason"], None
        )

    # Extract query details. Token counts assigned as zero if not found
    clean_response["total_tokens"] = get_nested_attr(
        response, ["usage", "total_tokens"], 0
    )
    clean_response["completion_tokens"] = get_nested_attr(
        response, ["usage", "completion_tokens"], 0
    )
    clean_response["prompt_tokens"] = get_nested_attr(
        response, ["usage", "prompt_tokens"], 0
    )
    clean_response["time_created"] = get_nested_attr(response, ["created"], None)
    clean_response["model"] = get_nested_attr(response, ["model"], None)

    return clean_response


if __name__ == "__main__":

    # Load headline data and remove attention check item
    headlines_df = pd.read_csv(HEADLINES_FILE)
    headlines_df = headlines_df[
        headlines_df.headline_fname != "attention_check.png"
    ].reset_index()
    assert len(headlines_df) == 40

    for prompt_type, prompt_info in PROMPTS.items():

        print(f"PROMPT TYPE: {prompt_type}")
        print("-" * 50)

        system_content = prompt_info["system_content"]
        user_content = prompt_info["user_content"]

        output_fname = f"fact_checks_{prompt_type}.jsonl"
        output_fp = os.path.join(OUTPUT_DIR, output_fname)

        with open(output_fp, "+a") as f_out:

            for idx, headline_row in headlines_df.iterrows():

                print(f"Working on headline #{idx}")

                user_content_w_headline = user_content.format(headline_row.text)
                if system_content is None:
                    response = call_no_instructions(prompt=user_content_w_headline)
                else:
                    response = call_json_w_instructions(
                        system_content, user_content_w_headline
                    )

                # Clean response and add information about the query
                clean_response = parse_response(response)
                clean_response.update(
                    {
                        "prompt_type": prompt_type,
                        "system_content": system_content,
                        "user_content": user_content_w_headline,
                        "veracity": headline_row.veracity,
                    }
                )

                # When we make a JSON output request, convert the JSON string
                # to a dictionary before saving for cleaner data
                if system_content is not None:
                    try:
                        clean_response["response_text"] = dict(
                            json.loads(clean_response["response_text"])
                        )
                    except JSONDecodeError as e:
                        print(f"\tJSONDecodeError encountered!")
                        print(f"\tHeadline: {headline_row.text}\n")

                    except Exception as e:
                        print(f"\tUnknown error encountered!")
                        print(f"\tHeadline: {headline_row.text}\n")

                f_out.write(f"{json.dumps(clean_response)}\n")

    print("--- Script complete ---")
