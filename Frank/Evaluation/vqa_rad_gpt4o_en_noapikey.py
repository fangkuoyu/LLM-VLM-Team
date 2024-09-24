###############################################################################
# Visual Question Answering on VQA-RAD by GPT4o with four evaluation methods:
#
# - BERTScore F1
# - G-Eval 
# - RAGAS correctness
# - RAGAS similarity
# 
# Created: 2024/09/12
# Updated: 2024/09/24
# Author:  Frank Yu
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

import os
import base64
import mimetypes

from datasets import Dataset 
from datasets import load_dataset

from evaluate import load

from ragas import evaluate
from ragas.metrics import answer_correctness, answer_similarity

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# load data -------------------------------------------------------------------
print(">>> flaviagiammarino/vqa-rad ")

data = load_dataset("flaviagiammarino/vqa-rad")

print(data)
print()

print(data.shape)
print()

# define image process --------------------------------------------------------
def image_to_base64(image_path):
    # Guess the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    
    if not mime_type or not mime_type.startswith('image'):
        raise ValueError("The file type is not recognized as an image")
    
    # Read the image binary data
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Format the result with the appropriate prefix
    image_base64 = f"data:{mime_type};base64,{encoded_string}"
    
    return image_base64

print(">>> connect OpenAI ---------------------------------------------------")
os.environ["OPENAI_API_KEY"] = "your api key"

client_openai = OpenAI()

final_table = pd.DataFrame(columns=['Question',
                                    'Ground Truth',
                                    'Prediction',
                                    'RAGAS correctness',
                                    'RAGAS similarity',
                                    'BERT score',
                                    'GEval score',
                                    'GEval reason'])

print(">>> loop -------------------------------------------------------------")
for idx in range(50):
    print(">>> index = ", idx)
    
    row = data['train'][idx]
    
    plt.imshow(row['image'])
    file_path = str(idx) + '.jpg'
    plt.savefig(file_path)
    
    question = "Answering the question as brief as possible:" + row['question']
    ground_truth = row['answer']
    
    base64_string = image_to_base64(file_path)

    response = client_openai.chat.completions.create(
                    model = "gpt-4o",
                    messages=[
                                {
                                    "role": "user",
                                    "content": [
                                                {"type": "text", "text": question},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                                    "url": base64_string,
                                                                    "detail": "low"
                                                                    }
                                                },
                                                ],
                                }
                                ],
                    max_tokens=1000,
                    )

    prediction = response.choices[0].message.content
    
    # calculate RAGAS correctness and similarity ------------------------------
    data_samples = {
                    'question': [question],
                    'answer': [prediction],
                    'ground_truth': [ground_truth]
                    }
    dataset = Dataset.from_dict(data_samples)
    ragas_score = evaluate(dataset, metrics=[answer_correctness, answer_similarity])
    
    # calculate BERTScore -----------------------------------------------------
    bertscore = load("bertscore")
    bert_score = bertscore.compute(predictions=[prediction], references=[ground_truth], lang="en")

    # calculate G-Eval --------------------------------------------------------
    test_case = LLMTestCase(input = question,
                            actual_output = prediction,
                            expected_output = ground_truth)
    correctness_metric = GEval( name="Correctness",
                                criteria="Correctness - determine if the actual output is correct according to the expected output.",
                                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
                                )
    correctness_metric.measure(test_case)    

    # print result ------------------------------------------------------------
    print("Question:      ", question)
    print("Ground_Truth:  ", ground_truth)
    print("Prediction:    ", prediction)
    print("RAGAS Score:   ", ragas_score) 
    print("BERT Score:    ", bert_score)
    print("G-Eval Score:  ", correctness_metric.score)
    print("G-Eval Reason: ", correctness_metric.reason)
    print()
    
    # show diagram ------------------------------------------------------------
    """
    summary = "Question: " + question + "\n \n" + \
              "Ground Truth: " + ground_truth + "\n \n" + \
              "Prediction: " + prediction + "\n \n" + \
              "--------------" + "\n \n" + \
              "RAGAS Correctness: " + str("%.2f" % (100 * ragas_score['answer_correctness'])) + "\n \n" + \
              "RAGAS Similarity: "  + str("%.2f" % (100 * ragas_score['answer_similarity'])) + "\n \n" + \
              "BERTScore F1: "      + str("%.2f" % (100 * bert_score['f1'][0])) + "\n \n" + \
              "G-Eval Score: "      + str("%.2f" % (100 * correctness_metric.score)) + "\n \n" + \
              "G-Eval Reason: "     + correctness_metric.reason
    
    w = row['image'].size[0]
    h = row['image'].size[1]
    d = max([w,h])
    
    image_alignment = np.full((d, d, 3), 255)
    
    image_alignment[0:h, 0:w, 0:3] = row['image']
    
    plt.axis('off')
    
    plt.imshow(image_alignment)
    
    txt = plt.text(d + 30, 10, summary, wrap=True, fontsize=12,
                   horizontalalignment='left',
                   verticalalignment='top',)
    txt._get_wrap_line_width = lambda : 400
    
    plt.show()
    """
    
    # save into the final table -----------------------------------------------
    ragas_correctness = str("%.2f" % (100 * ragas_score['answer_correctness']))
    ragas_similarity  = str("%.2f" % (100 * ragas_score['answer_similarity']))
    bertscore_f1      = str("%.2f" % (100 * bert_score['f1'][0]))
    geval_score       = str("%.2f" % (100 * correctness_metric.score))
    geval_reason      = correctness_metric.reason
    
    final_table.loc[len(final_table)] = [question,
                                         ground_truth,
                                         prediction,
                                         ragas_correctness,
                                         ragas_similarity,
                                         bertscore_f1,
                                         geval_score,
                                         geval_reason]
    
final_table.to_excel('vqa_rad_gpt4o_en_final_table_50.xlsx')