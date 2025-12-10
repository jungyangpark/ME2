import os
import time
import re
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

import evaluate
import random
import numpy as np
from loguru import logger as eval_logger
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AzureOpenAI, OpenAI
import urllib.error

API_URL = os.getenv("AZURE_ENDPOINT", "END_POINT")
API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")

MAX_TRIAL = 3

# Prompt for GPT Evaluation
GPT_TEXT_EVAL_PROMPT = """
You are evaluating the quality of an AI-generated explanation for a math problem involving geometry or graph-based reasoning.

You will be given two texts:

1. A reference explanation written by a human teacher.
2. An AI-generated explanation written by a model.

Your task is to compare the two explanations and assess how accurately and effectively the AI-generated explanation captures the key geometric concepts and reasoning presented in the reference.

Please evaluate the model's explanation and provide four scores based on the criteria below:

---
### Scoring Criteria

1. Correctness  
   - Does the reasoning presented by the model make sense and help solve the problem appropriately?  
   - Rate on a Likert scale: **1, 2, 3, 4, or 5**

2. Reference Alignment  
   - Does the model follow the same logical reasoning and intent as the reference explanation, even if the wording differs?  
   - Rate on a Likert scale: **1, 2, 3, 4, or 5**

3. Use of Key Visual Elements  
   - Does the AI explanation refer to the same critical visual components (e.g., points, lines, angles, shapes) as the reference?  
   - Alternative terminology is acceptable if it clearly refers to the same element or serves the same purpose.  
   - Rate on a Likert scale: **1, 2, 3, 4, or 5**

---
### Output Format

Important: Report your rating using the exact format below:

{{"rating": [x, y, z]}}
  
— where 'x' is your score for correctness, 'y' for reference alignment, and 'z' for use of visual elements.

"""

# evaluation metrics
nltk.download = lambda *args, **kwargs: None
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

# Utils
def normalize_MCQA_model_response(answer):
    if isinstance(answer, str):
        extraction = answer.strip()
    else:
        try:
            extraction = str(answer)
        except:
            extraction = ""

    # extract "A" from "(A) text"
    letter = re.findall(r"\(([a-eA-E])\)", extraction)
    # extract "A" from "{A} text"
    letter = re.findall(r"\{([a-eA-E])\}", extraction)
    if len(letter) > 0:
        extraction = letter[0].upper()

    if len(extraction) > 1:
        extraction = extraction[0]

    return extraction

# Make prompt for problem
def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [doc["problem_image"].convert("RGB")]

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    task_num = lmms_eval_specific_kwargs["task"]
    text = ""
    if task_num == 0:
        # problem solving task
        text = "You are a math solver. For the problem below, **your task is ONLY to output the final answer** in one line."
        text += "\n**Do NOT provide any explanation, steps, or clarification. Just write the answer.**"        
        text += f"\n\nProblem: {doc['problem_text']}"
        text += "\nIMPORTANT!! Your final response must be formatted as: 'The final Answer is: {correct_answer}'"

    elif task_num == 1:
        # figure captioning task
        text = "\n\nYou should choose a set of visual elements from the multiple-choice options (A, B, C, D, or E) that best reflect how a teacher would visually guide a student to understand and solve the problem."
        text += f"\n\nProblem: \n{doc['problem_text']} \n\nAnswer: {doc['answer_text']}"
        text += f"\nThe solution process for the problem is as follows: \n\n{doc['summary_solution_text']}"
        text += f"\n\n{doc['visual_identification_option']}"

        text += "\n\nBased on this reasoning guidance, select **only one** of the option (A, B, C, D, or E) whose visual elements would be most helpful for students in understanding the problem and its solution."
        text += "\n\nThink carefully about how the selected visual elements support the reasoning process. "
        text += "You may briefly explain your thinking, but your response **must end** with the following format:"
        text += "\nThe final answer is: {A, B, C, D, or E}"
        text += "\nIMPORTANT!! Your final response must END with the format."

    elif task_num == 2:
        # solution generation task
        text = doc["problem_text"]
        text += f"\n\n### Answer: {doc['answer_text']} ###"
        chapter_title = doc["chapter_title"]
        section_title = doc["section_title"]
        sol_img_cap = "\n\n### Difference between the original image and the solution image ###\n"
        caption_text = []
        for item in doc["visual_key_points"]:
            for key, value in item.items():
                if value != None:
                    caption_text.append(f"{key}: {value}")
        sol_img_cap += "\n".join(caption_text)

        text += sol_img_cap
        text += "\n\nYou are a math teacher helping students understand how to solve problems clearly and effectively."
        text += "\nGiven a problem description, problem image and a list of key elements introduced or highlighted in the solution image, write an educational explanation that helps students."
        text += f"\nAdditionally, this problem is a problem of {chapter_title}/{section_title} chapter. You should explain the problem in the context of the chapter and section."
        text += "\nMake sure to reference both the original components from the problem image and any new annotations, highlights, or added elements from the solution image to enhance understanding."
        text += "\n\n### OUTPUT Example:"
        text += "{\n  \"solution_text\": \"\"\n}"

    return text

# Problem solving task(Toy-Task)
GPT_TOY_TASK_EVAL_PROMPT = """
You are evaluating whether a model's response to a math problem is correct. Each example contains:

- A ground truth answer (GT answer)
- A model-generated response (Model prediction)

### Instructions:
- Return '1' if the numeric value in the model response is equal to the ground truth, even if units or format differ (e.g., "13" vs "13 meters").
- Return '0' otherwise.


### Output Format
Important: Report your rating using the exact format below

{{Correctness: x}}

- where x is your score for correctness, rated on 0 or 1

### Examples:
GT answer: 42 meters
Model prediction: 42
-> Correctness: 1

GT answer: 4 meter
Model prediction: 4000 mm
-> Correctness: 1

GT answer: 20
Model prediction: 25
-> Correctness: 0
"""

def ME2_process_results_problem_solving(doc, result):
    """
    Evaluation code for toy task (Problem solving task)
    """
    reference = doc["answer_text"]
    predictions = result[0]
    print(predictions)
    # if "The answer is" in predictions:
    #     predictions = predictions.split("The answer is")[1].strip()
    question_type = doc["question_answer_type"]

    # Ask GPT to compare the answer
    if "The final Answer is:" in predictions:
        predictions = predictions.split("The final Answer is:")[1].strip()

    if question_type == "geometry_choice" or question_type == "graph_choice":
        normalized_predictions = normalize_MCQA_model_response(predictions)
        if normalized_predictions.strip() == str(reference).strip():
            score = 1
        else:
            score = 0

    elif question_type == "geometry_short" or question_type == "graph_short":
        client = AzureOpenAI(api_key=API_KEY, azure_endpoint=API_URL, api_version=API_VERSION)
        prompt = f"{GPT_TOY_TASK_EVAL_PROMPT}\n\nGT answer: {reference}\n## Model prediction: {predictions}(think aloud before generating the scores)"
        for attempt in range(MAX_TRIAL):
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.0
            }
            try:
                response = client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content.strip()

                score = response_text.split("{{Correctness:")[1].replace("}}", "").strip()
                score = int(score)
                break
            except urllib.error.HTTPError as error:
                if attempt < MAX_TRIAL and error.code == 429:
                    attempt -= 1
                    time.sleep(15)
                else:
                    raise error

            except Exception as e:
                error_msg = str(e)
                prompt += "Error occured during parsing your answer. This is your response. Becareful of the error and try again."

                eval_logger.error(f"Parsing Step: Attempt {attempt+1}/{MAX_TRIAL} failed with error: {error_msg}")
                eval_logger.error(f"GPT response: {response_text}")

                if attempt == MAX_TRIAL - 1:
                    score = 0

    list_toy_task = [0,0]
    if question_type == "geometry_choice" or question_type == "geometry_short":
        list_toy_task[0] = score
    elif question_type == "graph_choice" or question_type == "graph_short":
        list_toy_task[1] = score
    
    data_dict = {
        "toy_task": score,
        "toy_task_GE": list_toy_task[0],
        "toy_task_GR": list_toy_task[1]
    }

    return data_dict

################################################################################################################
# Captioning task v2(MCQA)
def ME2_process_results_captioning(doc, result):
    reference = doc["visual_identification_answer"]
    predictions = result[0]
    if "The answer is" in predictions:
        predictions = predictions.split("The answer is")[1].strip()

    question_type = doc["question_answer_type"]

    if "The final answer is:" in predictions:
        predictions = predictions.split("The final answer is:")[-1].strip()

    normalized_predictions = normalize_MCQA_model_response(predictions)
    if normalized_predictions.strip() == str(reference).strip():
        score = 1
    else:
        score = 0

    list_captioning_MCQA = [0,0]
    if "geometry" in question_type:
        list_captioning_MCQA[0] = score
    elif "graph" in question_type:
        list_captioning_MCQA[1] = score
    
    data_dict = {
        "Caption_MCQA": score,
        "Caption_MCQA_GE": list_captioning_MCQA[0],
        "Caption_MCQA_GR": list_captioning_MCQA[1]
    }

    return data_dict

################################################################################################################
# Solution text task
def get_gpt_score_solution(predictions, reference, answer):
    """
        Returns the score of the solution text task evaluated by GPT.
    """
    client = AzureOpenAI(api_key=API_KEY, azure_endpoint=API_URL, api_version=API_VERSION)

    correctnesses = []
    reference_alignments = []
    visual_references = []
    mathematical_accuracies = []
    for pred, ref, ans in zip(predictions, reference, answer):
        prompt = f"{GPT_TEXT_EVAL_PROMPT}\n\n### GT Explanation\n\n{ref}\n\n###Model Explanation \n\n{pred}\n\n###Answer: {ans}\n\n(think aloud before generating the scores)"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.01
        }

        for attempt in range(MAX_TRIAL):
            try:
                response = client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content.strip()
                ratings = response_text.split('"rating": [')[-1].split("]")[0].split(", ")

                correctness = float(ratings[0].strip())
                reference_alignment = float(ratings[1].strip())
                visual_reference = float(ratings[2].strip())
                break

            except urllib.error.HTTPError as error:
                if attempt < MAX_TRIAL and error.code == 429:
                    attempt -= 1
                    time.sleep(15)
                else:
                    raise error

            except Exception as e:
                error_msg = str(e)
                eval_logger.error(f"Text Scoring step: Attempt {attempt+1}/{MAX_TRIAL} failed with error: {error_msg}")
                eval_logger.error(f"GPT response: {response_text}")
                if attempt == MAX_TRIAL - 1:
                    eval_logger.error(f"All {MAX_TRIAL} attempts failed. Last error: {error_msg}")
                    correctness = 0
                    reference_alignment = 0
                    visual_reference = 0

        correctnesses.append(correctness)
        reference_alignments.append(reference_alignment)
        visual_references.append(visual_reference)
    return {
        "correctness": np.mean(correctnesses),
        "reference_alignment": np.mean(reference_alignments),
        "visual_reference": np.mean(visual_references),
    }

def get_score_dict_solution_text(predictions, reference, answer, question_type):
    # bleu-4 score
    bleu1 = bleu.compute(predictions=[predictions], references=[reference], max_order=1)
    bleu2 = bleu.compute(predictions=[predictions], references=[reference], max_order=2)
    bleu3 = bleu.compute(predictions=[predictions], references=[reference], max_order=3)
    bleu4 = bleu.compute(predictions=[predictions], references=[reference], max_order=4)

    rouge_score = rouge.compute(predictions=[predictions], references=[reference])
    meteor_score = meteor.compute(predictions=[predictions], references=[reference])

    bertscore_score = bertscore.compute(predictions=[predictions], references=[reference], lang="en")

    gpt_score = get_gpt_score_solution(predictions=[predictions], reference=[reference], answer=[answer])
    list_correctness_gpt_score = [0,0,0,0]
    list_correctness_gpt_score[question_type] = gpt_score["correctness"]
    list_reference_alignment_gpt_score = [0,0,0,0]
    list_reference_alignment_gpt_score[question_type] = gpt_score["reference_alignment"]
    list_visual_reference_gpt_score = [0,0,0,0]
    list_visual_reference_gpt_score[question_type] = gpt_score["visual_reference"]

    list_bleu1_score = [0,0]
    list_bleu2_score = [0,0]
    list_bleu3_score = [0,0]
    list_bleu4_score = [0,0]
    list_rouge1_score = [0,0]
    list_rouge2_score = [0,0]
    list_rougeL_score = [0,0]
    list_rougeLsum_score = [0,0]
    list_meteor_score = [0,0]
    list_bertscore_score_precision = [0,0]
    list_bertscore_score_recall = [0,0]
    list_bertscore_score_f1 = [0,0]

    if question_type == 0 or question_type == 1:
        list_bleu1_score[0] = bleu1['bleu']
        list_bleu2_score[0] = bleu2['bleu']
        list_bleu3_score[0] = bleu3['bleu']
        list_bleu4_score[0] = bleu4['bleu']
        list_rouge1_score[0] = rouge_score["rouge1"]
        list_rouge2_score[0] = rouge_score["rouge2"]
        list_rougeL_score[0] = rouge_score["rougeL"]
        list_rougeLsum_score[0] = rouge_score["rougeLsum"]
        list_meteor_score[0] = meteor_score["meteor"]
        list_bertscore_score_precision[0] = bertscore_score["precision"][0]
        list_bertscore_score_recall[0] = bertscore_score["recall"][0]
        list_bertscore_score_f1[0] = bertscore_score["f1"][0]

    elif question_type == 2 or question_type == 3:
        list_bleu1_score[1] = bleu1['bleu']
        list_bleu2_score[1] = bleu2['bleu']
        list_bleu3_score[1] = bleu3['bleu']
        list_bleu4_score[1] = bleu4['bleu']
        list_rouge1_score[1] = rouge_score["rouge1"]
        list_rouge2_score[1] = rouge_score["rouge2"]
        list_rougeL_score[1] = rouge_score["rougeL"]
        list_rougeLsum_score[1] = rouge_score["rougeLsum"]
        list_meteor_score[1] = meteor_score["meteor"]
        list_bertscore_score_precision[1] = bertscore_score["precision"][0]
        list_bertscore_score_recall[1] = bertscore_score["recall"][0]
        list_bertscore_score_f1[1] = bertscore_score["f1"][0]

    data_dict = {
        "Rouge_1": rouge_score["rouge1"],
        "Rouge_2": rouge_score["rouge2"],
        "Rouge_L": rouge_score["rougeL"],
        "Rouge_Lsum": rouge_score["rougeLsum"],
        "Bleu1": bleu1['bleu'],
        "Bleu2": bleu2['bleu'],
        "Bleu3": bleu3['bleu'],
        "Bleu4": bleu4['bleu'],
        "Meteor": meteor_score["meteor"],
        "BertScore_precision": bertscore_score["precision"][0],
        "BertScore_recall": bertscore_score["recall"][0],
        "BertScore_f1": bertscore_score["f1"][0],

        "Correctness": gpt_score["correctness"],
        "Reference_alignment": gpt_score["reference_alignment"],
        "Visual_reference": gpt_score["visual_reference"],

        # Sub scores
        # GPT score
        "Correctness_GEC": list_correctness_gpt_score[0],
        "Correctness_GES": list_correctness_gpt_score[1],
        "Correctness_GRC": list_correctness_gpt_score[2],
        "Correctness_GRS": list_correctness_gpt_score[3],
        "Reference_alignment_GEC": list_reference_alignment_gpt_score[0],
        "Reference_alignment_GES": list_reference_alignment_gpt_score[1],
        "Reference_alignment_GRC": list_reference_alignment_gpt_score[2],
        "Reference_alignment_GRS": list_reference_alignment_gpt_score[3],
        "Visual_reference_GEC": list_visual_reference_gpt_score[0],
        "Visual_reference_GES": list_visual_reference_gpt_score[1],
        "Visual_reference_GRC": list_visual_reference_gpt_score[2],
        "Visual_reference_GRS": list_visual_reference_gpt_score[3],

        # Traditional metrics
        "Bleu1_GE": list_bleu1_score[0],
        "Bleu2_GE": list_bleu2_score[0],
        "Bleu3_GE": list_bleu3_score[0],
        "Bleu4_GE": list_bleu4_score[0],
        "Rouge_1_GE": list_rouge1_score[0],
        "Rouge_2_GE": list_rouge2_score[0],
        "Rouge_L_GE": list_rougeL_score[0],
        "Rouge_Lsum_GE": list_rougeLsum_score[0],
        "Meteor_GE": list_meteor_score[0],
        "BertScore_precision_GE": list_bertscore_score_precision[0],
        "BertScore_recall_GE": list_bertscore_score_recall[0],
        "BertScore_f1_GE": list_bertscore_score_f1[0],

        "Bleu1_GR": list_bleu1_score[1],
        "Bleu2_GR": list_bleu2_score[1],
        "Bleu3_GR": list_bleu3_score[1],
        "Bleu4_GR": list_bleu4_score[1],
        "Rouge_1_GR": list_rouge1_score[1],
        "Rouge_2_GR": list_rouge2_score[1],
        "Rouge_L_GR": list_rougeL_score[1],
        "Rouge_Lsum_GR": list_rougeLsum_score[1],
        "Meteor_GR": list_meteor_score[1],
        "BertScore_precision_GR": list_bertscore_score_precision[1],
        "BertScore_recall_GR": list_bertscore_score_recall[1],
        "BertScore_f1_GR": list_bertscore_score_f1[1],
    }
    return data_dict

def ME2_process_results_solution(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    predictions = result[0]
    # solution generation task
    reference = doc["solution_text"]
    answer = doc["answer_text"]

    question_type = doc["question_answer_type"]
    if question_type == "geometry_choice":
        data_dict = get_score_dict_solution_text(predictions, reference, answer, 0)
    elif question_type == "geometry_short":
        data_dict = get_score_dict_solution_text(predictions, reference, answer, 1)
    elif question_type == "graph_choice":
        data_dict = get_score_dict_solution_text(predictions, reference, answer, 2)
    elif question_type == "graph_short":
        data_dict = get_score_dict_solution_text(predictions, reference, answer, 3)

    return data_dict

# Returns the average of the results
def ME2_aggregation_result(results, metric, args):
    result = np.mean(results)
    eval_logger.info(f"[{metric}]: {result}")
    
    if metric == "Rouge_1": return result
    elif metric == "Rouge_2": return result
    elif metric == "Rouge_L": return result
    elif metric == "Rouge_Lsum": return result
    elif metric == "Bleu1": return result
    elif metric == "Bleu2": return result
    elif metric == "Bleu3": return result
    elif metric == "Bleu4": return result
    elif metric == "Meteor": return result
    elif metric == "BertScore_precision": return result
    elif metric == "BertScore_recall": return result
    elif metric == "BertScore_f1": return result

    elif metric == "Correctness": return result
    elif metric == "Reference_alignment": return result
    elif metric == "Visual_reference": return result

    elif metric == "Correctness_GEC": return result
    elif metric == "Correctness_GES": return result
    elif metric == "Correctness_GRC": return result
    elif metric == "Correctness_GRS": return result
    elif metric == "Reference_alignment_GEC": return result
    elif metric == "Reference_alignment_GES": return result
    elif metric == "Reference_alignment_GRC": return result
    elif metric == "Reference_alignment_GRS": return result
    elif metric == "Visual_reference_GEC": return result
    elif metric == "Visual_reference_GES": return result
    elif metric == "Visual_reference_GRC": return result
    elif metric == "Visual_reference_GRS": return result
    
    # Sub scores
    elif metric == "Rouge_1_GE": return result
    elif metric == "Rouge_2_GE": return result
    elif metric == "Rouge_L_GE": return result
    elif metric == "Rouge_Lsum_GE": return result
    elif metric == "Rouge_1_GR": return result
    elif metric == "Rouge_2_GR": return result
    elif metric == "Rouge_L_GR": return result
    elif metric == "Rouge_Lsum_GR": return result

    elif metric == "Bleu1_GE": return result
    elif metric == "Bleu2_GE": return result
    elif metric == "Bleu3_GE": return result
    elif metric == "Bleu4_GE": return result
    elif metric == "Bleu1_GR": return result
    elif metric == "Bleu2_GR": return result
    elif metric == "Bleu3_GR": return result
    elif metric == "Bleu4_GR": return result

    elif metric == "Meteor_GE": return result
    elif metric == "Meteor_GR": return result

    elif metric == "BertScore_precision_GE": return result
    elif metric == "BertScore_recall_GE": return result
    elif metric == "BertScore_f1_GE": return result
    elif metric == "BertScore_precision_GR": return result
    elif metric == "BertScore_recall_GR": return result
    elif metric == "BertScore_f1_GR": return result

    # Toy task
    elif metric == "ps": return result
    elif metric == "ps_GE": return result
    elif metric == "ps_GR": return result

    # Captioning task MCQA
    elif metric == "Caption_MCQA": return result
    elif metric == "Caption_MCQA_GE": return result
    elif metric == "Caption_MCQA_GR": return result

def ME2_rouge1(results, args): return ME2_aggregation_result(results, "Rouge_1", args)
def ME2_rouge2(results, args): return ME2_aggregation_result(results, "Rouge_2", args)
def ME2_rougeL(results, args): return ME2_aggregation_result(results, "Rouge_L", args)
def ME2_rougeLsum(results, args): return ME2_aggregation_result(results, "Rouge_Lsum", args)
def ME2_bleu1(results, args): return ME2_aggregation_result(results, "Bleu1", args)
def ME2_bleu2(results, args): return ME2_aggregation_result(results, "Bleu2", args)
def ME2_bleu3(results, args): return ME2_aggregation_result(results, "Bleu3", args)
def ME2_bleu4(results, args): return ME2_aggregation_result(results, "Bleu4", args)
def ME2_meteor(results, args): return ME2_aggregation_result(results, "Meteor", args)
def ME2_bertscore_precision(results, args): return ME2_aggregation_result(results, "BertScore_precision", args)
def ME2_bertscore_recall(results, args): return ME2_aggregation_result(results, "BertScore_recall", args)
def ME2_bertscore_f1(results, args): return ME2_aggregation_result(results, "BertScore_f1", args)

# GPT score of solution text task
def ME2_correctness(results, args): return ME2_aggregation_result(results, "Correctness", args)
def ME2_reference_alignment(results, args): return ME2_aggregation_result(results, "Reference_alignment", args)
def ME2_visual_reference(results, args): return ME2_aggregation_result(results, "Visual_reference", args)

# Sub scores
def ME2_correctness_GEC(results, args): return ME2_aggregation_result(results, "Correctness_GEC", args)
def ME2_correctness_GES(results, args): return ME2_aggregation_result(results, "Correctness_GES", args)
def ME2_correctness_GRC(results, args): return ME2_aggregation_result(results, "Correctness_GRC", args)
def ME2_correctness_GRS(results, args): return ME2_aggregation_result(results, "Correctness_GRS", args)

def ME2_reference_alignment_GEC(results, args): return ME2_aggregation_result(results, "Reference_alignment_GEC", args)
def ME2_reference_alignment_GES(results, args): return ME2_aggregation_result(results, "Reference_alignment_GES", args)
def ME2_reference_alignment_GRC(results, args): return ME2_aggregation_result(results, "Reference_alignment_GRC", args)
def ME2_reference_alignment_GRS(results, args): return ME2_aggregation_result(results, "Reference_alignment_GRS", args)

def ME2_visual_reference_GEC(results, args): return ME2_aggregation_result(results, "Visual_reference_GEC", args)
def ME2_visual_reference_GES(results, args): return ME2_aggregation_result(results, "Visual_reference_GES", args)
def ME2_visual_reference_GRC(results, args): return ME2_aggregation_result(results, "Visual_reference_GRC", args)
def ME2_visual_reference_GRS(results, args): return ME2_aggregation_result(results, "Visual_reference_GRS", args)

# ROUGE Sub scores
def ME2_rouge1_GE(results, args): return ME2_aggregation_result(results, "Rouge_1_GE", args)
def ME2_rouge2_GE(results, args): return ME2_aggregation_result(results, "Rouge_2_GE", args)
def ME2_rougeL_GE(results, args): return ME2_aggregation_result(results, "Rouge_L_GE", args)
def ME2_rougeLsum_GE(results, args): return ME2_aggregation_result(results, "Rouge_Lsum_GE", args)

def ME2_rouge1_GR(results, args): return ME2_aggregation_result(results, "Rouge_1_GR", args)
def ME2_rouge2_GR(results, args): return ME2_aggregation_result(results, "Rouge_2_GR", args)
def ME2_rougeL_GR(results, args): return ME2_aggregation_result(results, "Rouge_L_GR", args)
def ME2_rougeLsum_GR(results, args): return ME2_aggregation_result(results, "Rouge_Lsum_GR", args)

# BLEU Sub scores
def ME2_bleu1_GE(results, args): return ME2_aggregation_result(results, "Bleu1_GE", args)
def ME2_bleu2_GE(results, args): return ME2_aggregation_result(results, "Bleu2_GE", args)
def ME2_bleu3_GE(results, args): return ME2_aggregation_result(results, "Bleu3_GE", args)
def ME2_bleu4_GE(results, args): return ME2_aggregation_result(results, "Bleu4_GE", args)

def ME2_bleu1_GR(results, args): return ME2_aggregation_result(results, "Bleu1_GR", args)
def ME2_bleu2_GR(results, args): return ME2_aggregation_result(results, "Bleu2_GR", args)
def ME2_bleu3_GR(results, args): return ME2_aggregation_result(results, "Bleu3_GR", args)
def ME2_bleu4_GR(results, args): return ME2_aggregation_result(results, "Bleu4_GR", args)

# Meteor Sub scores
def ME2_meteor_GE(results, args): return ME2_aggregation_result(results, "Meteor_GE", args)
def ME2_meteor_GR(results, args): return ME2_aggregation_result(results, "Meteor_GR", args)

# BERTScore Sub scores
def ME2_bertscore_precision_GE(results, args): return ME2_aggregation_result(results, "BertScore_precision_GE", args)
def ME2_bertscore_recall_GE(results, args): return ME2_aggregation_result(results, "BertScore_recall_GE", args)
def ME2_bertscore_f1_GE(results, args): return ME2_aggregation_result(results, "BertScore_f1_GE", args)

def ME2_bertscore_precision_GR(results, args): return ME2_aggregation_result(results, "BertScore_precision_GR", args)
def ME2_bertscore_recall_GR(results, args): return ME2_aggregation_result(results, "BertScore_recall_GR", args)
def ME2_bertscore_f1_GR(results, args): return ME2_aggregation_result(results, "BertScore_f1_GR", args)

# Score of Problem Solving Toy task
def ME2_ps(results, args): return ME2_aggregation_result(results, "ps", args)
def ME2_ps_GE(results, args): return ME2_aggregation_result(results, "ps_GE", args)
def ME2_ps_GR(results, args): return ME2_aggregation_result(results, "ps_GR", args)

# Score of Problem Solving Toy task
def ME2_Caption_MCQA(results, args): return ME2_aggregation_result(results, "Caption_MCQA", args)
def ME2_Caption_MCQA_GE(results, args): return ME2_aggregation_result(results, "Caption_MCQA_GE", args)
def ME2_Caption_MCQA_GR(results, args): return ME2_aggregation_result(results, "Caption_MCQA_GR", args)
