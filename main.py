from scipy.misc import derivative
import json_lines
from transformers import(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
import torch
import QA2D_predict
import gec_predict
import numpy


# find the context that contains answer span.
def extract_context(start, end, passage):
    i = start
    j = end
    while(i > 0): # find the start of the sentence containing the answer
        if(passage[i] == '.'):
            i -= 1
            break
        i -= 1
    ii = 0
    if(i != 0): # there are still sentences before this sentence
        ii = i + 2
        while(i > 0):
            if(passage[i] == '.'):
                i += 1
                break
            i -= 1
    
    while(j < len(passage)): # find the end of the sentence containing the answer
        if(passage[j] == '.'):
            j += 1
            break
        j += 1
    jj = len(passage)-1
    if(j != len(passage)-1): # there are still sentences after this sentence
        jj = j
        while(j < len(passage)):
            if(passage[j] == '.'):
                break
            j += 1
    return passage[i:j], passage[ii:jj]


def main():
    MRC_QA_info = [] #store the contexts and questions of a quesiton
    generated_QA_pairs = []
    with open("./my_generated_questions.jsonl", "r", encoding="utf-8") as q:
        with open("my_passages_with_answers.jsonl", "r", encoding="utf-8") as p:
            with open("./gec_input/input.txt", "w", encoding="utf-8") as w:
                passage_info = json_lines.reader(p)
                passage = [p['passage'] for p in passage_info]
                QA_info = json_lines.reader(q)
                for item in QA_info:
                    w.write(item['question'])
                    w.write('\n')
                    question_info = {}
                    question_info['question'] = item['question']
                    question_info['context'], item['one_sentence_context'] = extract_context(item['metadata']['answer_start'], item['metadata']['answer_end'], passage[0])
                    MRC_QA_info.append(question_info)
                    item['context'] = question_info['context']
                    generated_QA_pairs.append(item)                   
                    
    '''
    GEC model to judge grammar score.
    '''                
    gec_results = gec_predict.gec()                
    for i in range(len(generated_QA_pairs)):
        generated_QA_pairs[i]['grammar_score'] = min(max(1-(gec_results[i]-1)/3, 0), 1) # the reason why "-1" is to ignore the '?' added by gec model

                
    '''
    adopt QA model to measure answerability score.
    '''
    QA_model_name = "deepset/roberta-base-squad2"
    QA_pipeline = pipeline('question-answering', model=QA_model_name, tokenizer=QA_model_name)
    QA_answers = QA_pipeline(numpy.array(MRC_QA_info))
    # print(QA_answers)
    # print()
    for i in range(len(QA_answers)):
        pred_answer = QA_answers[i]['answer']
        pred_score = QA_answers[i]['score']
        ori_answer = generated_QA_pairs[i]['answer']
        # exact match
        if(pred_answer == ori_answer):
            generated_QA_pairs[i]["answerability_score"] = 0.5 * pred_score + 0.5 * 1
        # inclusion relation
        elif((pred_answer in ori_answer) or (ori_answer in pred_answer)):
            generated_QA_pairs[i]["answerability_score"] = 0.5 * pred_score + 0.5 * 0.8
        else:
            generated_QA_pairs[i]["answerability_score"] = 0


    '''
    use QA2D model to generate declarative answer sentence. Then use NLI model to evaluate inference score.
    '''
    with open('QA2D_input/input.txt', 'w', encoding="utf-8") as writer:
        for item in generated_QA_pairs:
            writer.write("question: %s answer: %s\n" % (item['question'], item['answer']))

    QA2D_model_path = "MarkS/bart-base-qa2d"
    input_file = "./QA2D_input/input.txt"
    output_file = "./QA2D_output/output.txt"
    QA2D_predict.QA2D_transform(QA2D_model_path, input_file, output_file, batch_size=16)

    # then use NLI model to predict inference_score
    nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    nli_model.cuda()

    with open('QA2D_output/output.txt', 'r', encoding="utf-8") as reader:
        for item in generated_QA_pairs:
            declarative_answer = reader.readline()
            # NLI_input = item['context'] + " </s> " + declarative_answer
            # print(NLI_input)
            tokenized_inputs = nli_tokenizer(item['context'], declarative_answer, return_tensors="pt").to('cuda')
            with torch.no_grad():
                logits = nli_model(**tokenized_inputs).logits.to('cpu')
            scores = torch.softmax(logits, -1).numpy()
            item['inference_score'] = max(0, scores[0][2] + 0.2*scores[0][1] - scores[0][0])


    #culculate final score
    for item in generated_QA_pairs:
        item['final_score'] = item['grammar_score']*0.2 + item['answerability_score']*0.3 + item['inference_score']*0.5
        
    sorted_QA_pairs = sorted(generated_QA_pairs, key=lambda e:e.__getitem__('final_score'), reverse=True)#rerank the questions
    # print(sorted_QA_pairs)
    # the detailed information of each question is stored in sorted_QA_pairs
    with open('./final_result.txt', 'w', encoding="utf-8") as writer:
        for item in sorted_QA_pairs:
            temp = {}
            temp['final_score'] = format(item['final_score'], '.3f')
            temp['answerability_score'] = format(item['answerability_score'], '.3f')
            temp['inference_score'] = format(item['inference_score'], '.3f')
            temp['grammar_score'] = format(item['grammar_score'], '.3f')
            temp['question'] = item['question']
            temp['answer'] = item['answer']
            writer.write("%s\n" % str(temp))

if __name__ == '__main__':
    main()
        