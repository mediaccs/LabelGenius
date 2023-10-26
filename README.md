# LabelGenius - GPT-Powered Media Text Labeler
## A Tool for Dynamic and Flexible Labeling with GPT Models for Media Text Classification
This GitHub repository holds the code and materials designed for a dynamic and innovative approach to text classification using GPT-4 and GPT-3.5 models. It introduces an iterative refinement process, initiating the classification with GPT-4 and progressively enhancing accuracy through human validation and model fine-tuning.

The original replication code and codebook can be found in the replication folder ("example").

## Authors
- [Jiacheng Huang](https://github.com/JcHuang11234)
- [Chris Chao Su](https://github.com/chrischaosu)

## To cite this tool:
Huang, J., & Su, C.C. (2023). LabelGenius-GPT-Powered Media Text Labeler: A Tool for Dynamic and Flexible Labeling with GPT Models for Media Text Classification [Computer software]. https://github.com/mediaccs/LabelGenius

## Overview
### Iterative Refinement and Dynamic Labeling Process
This methodology is rooted in the iterative refinement and dynamic labeling process. It begins with GPT-4 undertaking the initial classification of a small sample of text. Human reviewers then validate this classification, ensuring accuracy. The validated data serves dual purposes - 1) If the classifications rendered by GPT-4 meet the established accuracy, the model can be applied to a larger sample for further validation and then to the entire dataset; 2) if the classification needs further refinement, the human-validated results will be used as dataset to fine-turn the GPT-3.5 model.

1. ##### Multi-Theme Classification
This tool can process both single-them and  multi-theme classification. It’s equipped to dissect a text and concurrently classify it into multiple themes. 

2. ##### Multi-Verification
This tool is equipped with a multi-verification method. This method ensures the robustness of classifications by sending out multiple queries to the OpenAI API, retrieving a variety of results, and comparing them. 

##### Number of verification (once_verify_num)
OpenAI's pricing for classification tasks is based on the total number of tokens processed, including both the tokens in the questions and the answers. The cost of each classification is calculated as the sum of the tokens in the question and answer, multiplied by their respective prices per token. Retries are usually less cost-efficient than higher numbers of once_verify_num, as they re-incur the cost of question tokens.

To minimize costs, it is essential to strategically set the once_verify_num. This parameter should be adjusted based on the specific requirements and complexities of each classification task to avoid inconclusive results and reduce the need for costly retries. For example, in a one-theme, two-class variable classification task, setting an appropriate once_verify_num =3 helps prevent an equal number of responses for each class, thereby avoiding inconclusive results.


#### Dynamic Labeling
The dynamic labeling process significantly reduces the initial labeled data requirements. It begins the journey with a sample of 200 labeled posts, progressively adding more labeled data in iterations. Each addition refines the model's accuracy, ensuring that the classification becomes more precise with each step. Based on our experience, the fine-turned model with 600-800 labeled posts will be sufficient to process complex multi-theme classification.

#### Iterative Refinement
Every iteration of classification and validation refines the model's understanding and accuracy. The process is flexible, adapting to the complexities and nuances of the text being classified. This adaptability ensures that the model is not static but is dynamic, learning, and evolving with each iteration.



## Installation

```python
# load packages 
from gpt_classify_function import *
```

## Authentication

To utilize this package, you need to access OpenAI's API, requiring an API key. Follow these steps to set it up:
1. Create an Account for OpenAI Platform (https://platform.openai.com/overview)
2. Click "View API Key" , then "Create new secret key"
3. Copy the key, and take note of the key as it will be needed to authenticate your requests.

```python
import os
openai.api_key = "YOUR-API-Key"
```

## Step 1: Annoate the task 
We can now go ahead and classify the texts

## Function: gpt_classifier.classify_df
## Parameters:
*column:* Name of the column containing media-text to be classified.
*prompt:* A string that guides the GPT model for classification.
*model_name:* Specifies which GPT model to use for classification.
*label_num:* Number of dimensions for the variable being classified.
*valid_values:* A list of valid answers for the classification question.
*temperature:* Controls the randomness of the model’s output.
*q_name:* Name of the question or variable being classified. (The result will be save as q_named classfication)
*max_verify_retry:* Maximum number of retries for classification if previous attempts are not converged.
*once_verify_num:* Number of results generated from GPT in each classification attempt.
    The once_verify_num parameter signifies how many distinct outcomes the GPT model produces within a single verification round. For instance, if       once_verify_num is set to 5, the system will generate five unique results for the same input prompt in a single attempt. These results are then      compared to deduce a consensus.
    To determine the definitive answer, the frequency of each unique result is calculated. If there's a unique maximum frequency, that result is         chosen. However, in cases where multiple results share the highest frequency, this indicates a lack of convergence. The system will then retry       the classification, generating a new set of results. These new results will be combined with the old ones, and the process is repeated until a       unique answer with the highest frequency is found or until the max_verify_retry limit is reached.

### Example 1 single theme variable
#### Q1: Is this news article related to immigration in the United States? 
Note: Irrelevant articles include immigration out of the US and immigration issues in countries other than the US. An article that covers a list of stories (e.g., the top 10 news stories of this month) is also considered not relevant.

1)	Yes

2)	No


```python
# Set up the prompt
prompt_Q1 = """Here's a news article headline. Please code it based on the following criteria:
    Q1. **Relevance to U.S. Immigration**: Is the headline relevant to U.S. immigration? 
    Note: Irrelevant articles include immigration out of the US and immigration issues in countries other than the US. 
    An article that covers a list of stories (e.g., the top 10 news stories of this month) is also considered not relevant.
    - Return <1> for "Yes"
    - Return <2> for "No"
    Please only return "1" or "2" and do not provide any other information."""
```

```python
# load the data
Q1_initial_sample = pd.read.excel('03_Q1_initial_sample.xlsx')

# Setup permaters
if __name__ == "__main__":
    column_name = "Post_Title"  # the column name for the text to be classified
    model_name = "gpt-4"  # the GPT model to use
    label_num = 1  # the number of dimension/theme of this variable
    valid_values = ['1', '2']  # the valid answers from this question
    temperature = 0.7  # temperature 
    q_name = "Q1"  # the name of this question/variable
    once_verify_num = 3  # the numbers of results generated from GPT 
    max_verify_retry = 5  # the number of retries if the result is not converged
    prompt = prompt_Q1 # prompt

    # Initialize GPT Classifier
    gpt_classifier = GPTClassifier()

    # Setup classification task
    classification_task = ClassificationTask(
        column=column_name,
        prompt=prompt,
        model_name=model_name,
        label_num=label_num,
        valid_values=valid_values,
        temperature=temperature,
        q_name=q_name,
        once_verify_num=once_verify_num,
        max_verify_retry=max_verify_retry
    )

    # Classify and save the result
    result_df_q1_first_try = gpt_classifier.classify_df(Q1_initial_sample, classification_task)
    
    # Save to Excel
    result_df_q1_first_try.to_excel("04_Q1_initial_sample_result.xlsx")

```

### Example 2: Multi-theme variable
#### Q3-1/2: What is the main theme of this news story?  
Note: Code up to two dominant themes. Consider the following headline as an example: “A caravan of grandparents are crossing the country to protest family separation in Texas.” This headline should be coded as both 8) Public opinion and 3) Family. Whichever theme that comes first in the headline should be entered as Theme1. Enter “99” if there is no theme identified for both Q3-1 and Q3-2. 

Economic consequences

Crime/safety

Family

Immigrant wellbeing

Culture/society

Politics

Legislation/regulation

Public opinion

None of the above 

For better performance, we convert Q3 to code if each of the topics exists within the headline.

#### This is an example of *multi-theme classfication*.



```python
prompt_Q3 = '''Here's a news article headline. Please label if it belongs to the following theme. 
            Return <1> if this headline belongs to these themes and return <0> if it does not belong to the themes.
            Please identify up to two dominant themes from the headline, which means you can have a max of 2 <1> in the answer you generated.
            You don't have to label two topic if you don't fint it apply. Just enter 0s.
            - Economic consequences: The story is about economic benefits or costs, or the costs involving immigration-related issues, including: Cost of mass deportation; Economic benefits of immigration (more tax revenue, cheap labor; Economic costs of immigration (taking jobs from Americans, immigrants using healthcare and educational services, overcrowding, housing concerns)
            - Crime/safety: The story is about threats to American's safety, including: Immigration described as a major cause of increased rates of crime, gangs, drug trafficking, etc; Immigrants described as law-breakers who deserve punishment; Immigration described as a threat to national security via terrorism
            - Family: The story is about the impact of immigration on families, including: Separating children from parents; Breaking up multi-generational families; Interfering with children's continued schooling
            - Immigrant wellbeing: This story is about the negative impact of the immigration process on immigrants, including: Prejudice and bias toward immigrants; Physical and/or mental health or safety of immigrants; Immigration policies described as violations of immigrants' civil rights and liberties; Immigration policies regarding illegal immigrants described as unfair to immigrants who have waited to become citizens the legal way
            - Culture/society: This story is about societal-wide factors or consequences related to immigration, including:; Immigration as a threat to American cultural identity, way of living, the predominance of English and Christianity, etc.; Immigrants as isolated from the rest of America, unable to assimilate into communities; Immigration as part of the celebrated history of immigration in America / America-as-melting-pot; Immigration policies as exemplars of society's immorality; Impact of immigration on a specific subculture/community in the US
            - Politics:The story is mainly about the political issues around immigration, including: Political campaigns and upcoming elections (e.g., using immigration as a wedge issue or motivating force to get people to the polls); Fighting between the Democratic and Republican parties, or politicians; One political party or one politician’s stance on immigration. Therefore, when the news headline mentions a politician’s name, it often indicates the theme of politics
            - Legislation/regulation: The story is about issues related to regulating immigration through legislation and other institutional measures: New immigration legislation being introduced/argued over; Flaws in current/old legislation; Enforcement of current legislation
            - Public opinion: The study is about the public’s, including a specific community’s, reactions to immigration-related issues, including: Public opinion polls; Protests; Social media backlash; Community outrage; Celebrity responses/protests
            Answer using the JSON formet, [<0>,<0>,<0>,<0>,<0>,<0>,<0>,<0>]. Do not provide any other information'''

```

```python
# Q3 initial sample 
Q3_initial_sample = pd.readl.excel("11_Q3_inital_sample.xlsx")

if __name__ == "__main__":
    # Setup parameters
    column_name = "Post_Title"  # the column name for the text to be classified
    model_name = "gpt-4"  # the GPT model to use
    label_num = 8  # the number of dimensions of this variable
    valid_values = ['0', '1']  # the valid answers from this question
    temperature = 0.7  # temperature 
    q_name = "Q3"  # the name of this question/variable
    once_verify_num = 3  # the numbers of results generated from GPT 
    max_verify_retry = 5  # the number of retries if the previous is not converged
    prompt = prompt_Q3

    # Initialize GPT Classifier
gpt_classifier = GPTClassifier()

    # Setup classification task
classification_task = ClassificationTask(
        column=column_name,
        prompt=prompt,
        model_name=model_name,
        label_num=label_num,
        valid_values=valid_values,
        temperature=temperature,
        q_name=q_name,
        once_verify_num=once_verify_num,
        max_verify_retry=max_verify_retry
    )

    # Classify and save the result
result_df_q3_first_try = gpt_classifier.classify_df(Q3_initial_sample, classification_task)
result_df_q3_first_try.to_excel("12_Q3_inital_sample_result.xlsx")


```

### Step 2: review if the results contains any rows that need human validation
which has error or can not converge after the max_verify_retry that needs human validation


```python
# example for simple-theme classfication
review_human_validation_rows(df=result_df_q1_first_try, 
                             text_column="Post_Title",   # the column name for the text to be classified
                             valid_values= ['1', '2'] ,   # the valid answers from this question
                             classification_columns=["Q1_1_classification"]) # the colomn names for the classfication results

```

```python
review_human_validation_rows(df=result_df_q3_first_try, 
                             text_column="Post_Title", # the column name for the text to be classified
                             valid_values= ['0', '1'] ,  # the valid answers from this question
                             classification_columns=["Q3_1_classification","Q3_2_classification","Q3_3_classification","Q3_4_classification","Q3_5_classification","Q3_6_classification","Q3_7_classification","Q3_8_classification"]) # the colomn names fir the classfication results

```

### Step 3: caculate the accuracy 

#### Method 1: maunally check row by row
This code will show the text that needs to be classfied and the classfication result

If the result is correct, clikc enter; if not, please enter the correct answer

```python
overall_accuracy, result_df =  manual_validation(result_df_q1_first_try, # the data frame
                      text_column="Post_Title",  # the column name for the text to be classified
                      num_dimensions=1, # the number of dimension/theme of this variable
                      column_names=["Q1_1_classification"], # the colomn name for the predicted results
                      valid_values= ['1', '2']) # the valid answers from this question
```

#### Method 2: auto check the accuracy 
    
If we already have the human verified result in excel or csv

```python
accuracy_Q1_first_try= compute_accuracy(df=result_df_q1_first_try, 
                                        predict_cols=["Q1_1_classification"], # the colomn name for predicted results
                                        actual_cols=["Q1"]) # the colomn name for acual results
accuracy_Q1_first_try
```

```python
# caculate accuarcy for whole test df
accuracies = {}

for i in range(1, 9):  # Loop from 2 to 8
    predict_col = f"Q3_{i}_classification"
    actual_col = f"Q3_{i}"
    
    accuracy = compute_accuracy(df=test_df, 
                                predict_cols=[predict_col], 
                                actual_cols=[actual_col])
    
    accuracies[actual_col] = accuracy

# Print the accuracies
print("Here's the accuracy for the Q3 for the test df:")
for col, acc in accuracies.items():
    print(f"Accuracy for {col}: {acc}")
```

If the accuracy meets predefined criteria, we can proceed to apply the classification model to a larger sample for validation and eventually extend its application to the entire dataset. 

If the accuracy is unsatisfactory, we can use the human-validated results to fine-tune the ChatGPT model 3.5 to enhance its performance.

```python
# Step 4 (Optional): finetune ChatGPT 3.5 model

```

```python
# finetune the df to train GPT
prepare_finetune_data(df=result_df_q3_first_try, 
                      text_column= "Post_Title", 
                      num_dimensions=8, 
                      column_names=['Q3_1','Q3_2','Q3_3', 'Q3_4', 'Q3_5', 'Q3_6','Q3_7', 'Q3_8'], 
                      prompt=prompt_Q3, 
                      output_file_name="13_Q3_finetune_1.jsonl")
```

```python
# upload the Fine-tune dataset
finetune_file_Q3_1 = openai.File.create(
    file=open("13_Q3_finetune_1.jsonl", "rb"),
    purpose='fine-tune'
)
finetune_file_Q3_1
```

```python
# process the Fine-tune 
# It is normal to see an error saying that the id is not ready yet
# Just wait several minutes and allow OpenAI to process the result
finetune_file_Q3_1 = openai.FineTuningJob.create(
    training_file=finetune_file_Q3_1.id,
    model="gpt-3.5-turbo"
)
### This step usually takes several hours, depending on the capacity of OpenAI, and the number of assignment before
### You will received an email from OpenAI saying that the finetune job is done
### After process, when runing the "fine_tuning_job" you will see "  "trained_tokens": XXXX (instead of null)"
```

```python
# Test with the fine-tuned model
if __name__ == "__main__":
    # Setup parameters
    column_name = "Post_Title"  # the column name for the text to be classified
    model_name = gpt_fintuned_Q3_1  # the GPT model to use
    label_num = 8  # the number of dimensions of this variable
    valid_values = ['0', '1']  # the valid answers from this question
    temperature = 0.7  # temperature 
    q_name = "Q3"  # the name of this question/variable
    once_verify_num = 3  # the numbers of results generated from GPT 
    max_verify_retry = 5  # the number of retries if the previous is not converged

    # Prompt setup
prompt =  prompt_Q3

    # Initialize GPT Classifier
gpt_classifier = GPTClassifier()

    # Setup classification task
classification_task = ClassificationTask(
        column=column_name,
        prompt=prompt,
        model_name=model_name,
        label_num=label_num,
        valid_values=valid_values,
        temperature=temperature,
        q_name=q_name,
        once_verify_num=once_verify_num,
        max_verify_retry=max_verify_retry
    )

    # Classify and save the result
result_df_q3_second_try = gpt_classifier.classify_df(Q3_second_sample, classification_task)
result_df_q3_second_try.to_excel("15_Q3_second_sample_result.xlsx")

```
