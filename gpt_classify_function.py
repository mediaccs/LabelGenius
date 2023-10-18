import sqlitedict
import time
import pandas as pd
import openai
import time
import json
import csv
import hashlib
import threading
from traceback import format_exc
from loguru import logger
from dataclasses import dataclass



@dataclass
class ClassificationQuestion:
    prompt: str
    model_name: str
    valid_values: list
    temperature: float
    q_name: str
    text: str = None
    label_num: int = 1
    max_verify_retry: int = 2

    def get_key(self):
        ls = [
            self.prompt,
            self.model_name,
            self.valid_values,
            self.temperature,
            self.q_name,
            self.text,
            self.label_num,
            self.max_verify_retry,
        ]
        ls = map(str, ls)
        return hashlib.md5("".join(ls).encode("utf-8")).hexdigest()


@dataclass
class ClassificationTask:
    column: str
    prompt: str
    model_name: str
    valid_values: list
    temperature: float
    q_name: str
    label_num: int = 1
    once_verify_num: int = 10
    max_verify_retry: int = 2

    def create_question(self, text):
        return ClassificationQuestion(
            prompt=self.prompt,
            text=text,
            model_name=self.model_name,
            label_num=self.label_num,
            valid_values=self.valid_values,
            temperature=self.temperature,
            q_name=self.q_name,
            max_verify_retry=self.max_verify_retry,
        )


class DBCache:
    def __init__(self) -> None:
        self.db = sqlitedict.SqliteDict("db.sqlite", autocommit=True)

    def add(self, q: ClassificationQuestion, res):
        key = q.get_key()
        self.db[key] = res

    def get(self, q: ClassificationQuestion):
        key = q.get_key()
        return self.db.get(key)


class MaxRetryException(Exception):
    pass





class GPTClassifier:
    def __init__(self) -> None:
        self.cache = DBCache()

    def fetch(self, messages, model_name, temperature=0.7, n=1):
        retries = 3
        response = None

        def worker():
            nonlocal response
            try:
                response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=temperature, n=n)
            except Exception as e:
                print(e)

        for _ in range(retries):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)  # Wait for 5 seconds

            if response:  # If the response is set, break out of the loop
                break
            else:
                print("Request timed out. Retrying...")
                time.sleep(2)

        return response

    def classify(self, q: ClassificationQuestion, n):
        messages = [{"role": "system", "content": q.prompt}, {"role": "user", "content": f"Post: {q.text}"}]
        response = self.fetch(messages, q.model_name, q.temperature, n=n)
        results_raw_ls = [item["message"]["content"].strip() for item in response["choices"]]
        logger.debug(f"Raw results: {results_raw_ls}")
        valid_results_ls = []
        for item in results_raw_ls:
            # Replace single quotes with double quotes
            item = item.replace("'", '"')
            # Remove angle brackets
            item = item.replace('<', '').replace('>', '')
            try:
                parsed_item = json.loads(item)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse item: {item}")
                continue
            if isinstance(parsed_item, int):
                res = [str(parsed_item)]
            elif isinstance(parsed_item, list):
                res = list(map(str, parsed_item))
            else:
                raise ValueError(f"Invalid result {parsed_item}")
            if len(res) == q.label_num and all([r in q.valid_values for r in res]):
                valid_results_ls.append(res)
        logger.debug(f"Valid results: {valid_results_ls}")
        return valid_results_ls

    def multi_verify(self, q: ClassificationQuestion, n, retry=1, last_frequency_dic=None):
        if res := self.cache.get(q):
            logger.info(f"Cache hit {res} {q.text}")
            return res

        classifications: list = self.classify(q, n=n)
        if not classifications:
            if retry >= q.max_verify_retry:
                raise MaxRetryException(f"Max try reached for {q.text}")
            logger.info(f"Try:{retry}, Failed to verify, next trying....  {q.text}")
            return self.multi_verify(q, n=n, retry=retry + 1)

        frequency_dic = last_frequency_dic or {}
        for attempt in classifications:
            key = "_".join(attempt)
            frequency_dic[key] = frequency_dic.get(key, 0) + 1
        sorted_classifications = sorted(frequency_dic.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Try:{retry} Sorted classifications: {sorted_classifications}")
        if len(sorted_classifications) == 1 or sorted_classifications[0][1] > sorted_classifications[1][1]:
            logger.success(f"Try:{retry} Verified as {sorted_classifications[0][0]} for {q.text}")
            res = sorted_classifications[0][0].split("_")
            self.cache.add(q, res)
            return res
        elif retry < q.max_verify_retry:
            logger.info(f"Try:{retry}, Failed to converge, next trying....  {q.text}")
            return self.multi_verify(q, n=n, retry=retry + 1)
        else:
            raise MaxRetryException(f"Max try reached for {q.text}")

    def classify_df(self, df: pd.DataFrame, task: ClassificationTask):
        dic = df.to_dict(orient="records")
        for idx, item in enumerate(dic):
            try:
                logger.info(f"Classifying {idx+1}/{len(dic)} {item[task.column]}")
                res_ls = self.multi_verify(task.create_question(item[task.column]), n=task.once_verify_num)
            except MaxRetryException as e:
                logger.error(f"MaxRetryException Failed to classify {item[task.column]}. {e}")
                res_ls = ["err"] * task.label_num
            except Exception as e:
                logger.exception(f"Failed to classify {item[task.column]}. {e}")
                res_ls = ["err"] * task.label_num
            for idx, res in enumerate(res_ls):
                item[f"{task.q_name}_{idx+1}_classification"] = res

        return pd.DataFrame(dic)

    
def prepare_finetune_data(df, text_column, num_dimensions, column_names, prompt, output_file_name):
    """
    Prepare data for fine-tuning in OpenAI's expected format.
    
    Args:
    - df (pd.DataFrame): DataFrame containing validated results.
    - text_column (str): The name of the column containing text data.
    - num_dimensions (int): Number of classification dimensions.
    - prompt (str): The instruction prompt.
    - output_file_name (str): The name of the output file where the processed data will be written.
    - column_names (list): List of column names for the assistant's response.
    
    Returns:
    - None. Writes the processed data to the specified output file.
    """
    
    assert num_dimensions == len(column_names), "Number of dimensions should match the length of column_names list."
    
    with open(output_file_name, "w") as f:
        for _, row in df.iterrows():
            # Convert the numbers in the assistant's response columns to lists of strings.
            assistant_content = [str(row[col]) if pd.notna(row[col]) else None for col in column_names]
            # Remove None values
            assistant_content = [val for val in assistant_content if val is not None]
            
            # Convert the list to a string representation
            assistant_content_str = str(assistant_content)
            
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": row[text_column]
                },
                {
                    "role": "assistant",
                    "content": assistant_content_str
                }
            ]
            
            f.write(json.dumps({"messages": messages}) + "\n")

            
def manual_validation(result_df, text_column, num_dimensions, column_names, valid_values):
    # Add accuracy column for each dimension and initialize with 0s
    for col_name in column_names:
        result_df[f'{col_name}_accuracy'] = 0 

    for index, row in result_df.iterrows():
        print(f"\nPost: {row[text_column]}")

        # Display all classifications at once
        predictions = {col_name: row[col_name] for col_name in column_names}
        print(f"Predicted Classifications: {predictions}")

        for col_name in column_names:
            human_input = input(f"Enter the correct classification for {col_name} or press Enter if correct ({', '.join(valid_values)}): ").strip()
            
            while human_input not in [""] + valid_values:
                print(f"Invalid input. Please enter {', '.join(valid_values)} or press Enter.")
                human_input = input(f"Enter the correct classification for {col_name} or press Enter if correct ({', '.join(valid_values)}): ").strip()

            if human_input == "" or human_input == str(row[col_name]):
                result_df.at[index, f"{col_name}_accuracy"] = 1
            else:
                result_df.at[index, col_name] = human_input

    total_accuracy_cols = sum([result_df[f'{col_name}_accuracy'].sum() for col_name in column_names])
    sample_size = len(result_df) * num_dimensions
    overall_accuracy = total_accuracy_cols / sample_size
    
    return overall_accuracy, result_df


def compute_accuracy(df, predict_cols, actual_cols):
    # Ensure predict_cols and actual_cols have the same length
    if len(predict_cols) != len(actual_cols):
        raise ValueError("Mismatch between provided column names for prediction and actual values.")

    def compare_rows(row):
        predicted = sorted([row[col] for col in predict_cols])
        actual = sorted([row[col] for col in actual_cols])

        return predicted == actual

    # Apply the compare function to each row
    df['accurate'] = df.apply(compare_rows, axis=1)

    # Calculate accuracy
    accuracy = df['accurate'].sum() / len(df)
    return accuracy



import pandas as pd

def review_human_validation_rows(df, text_column, valid_values, classification_columns):

    # Create a mask to check for "err" in any of the classification columns
    needs_validation_mask = False
    for col in classification_columns:
        needs_validation_mask = needs_validation_mask | (df[col] == "err")

    # If no rows need validation
    if not needs_validation_mask.any():
        print("No row needs human validation, please proceed")
        return df

    # Filter rows that need human validation
    rows_needing_validation = df[needs_validation_mask]

    # Loop through these rows and prompt the user for valid input
    for idx, row in rows_needing_validation.iterrows():
        print(f"Reviewing text: {row[text_column]}")
        
        # Print all classification results for the row, regardless of their values
        for col in classification_columns:
            print(f"{col}: {row[col]}")
            
        # Prompt the user for valid input for columns with "err"
        for col in classification_columns:
            if row[col] == "err":                
                classification = input(f"Please provide a valid classification for {col} from {valid_values}: ")

                # Ensure the provided classification is in the list of valid values
                while classification not in valid_values:
                    print("Invalid input!")
                    classification = input(f"Please provide a valid classification for {col} from {valid_values}: ")

                # Update the dataframe with the new classification
                df.at[idx, col] = classification

    return df

