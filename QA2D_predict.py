from transformers import BartTokenizer, BartForConditionalGeneration
import torch

QA2D_tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
device = torch.device("cuda")
QA2D_results = []

def get_response(model, input_text, max_length=64, cuda=True):
    input = QA2D_tokenizer(input_text, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
    input = input.to(device)
    output = model.generate(input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            max_length=max_length,
                            num_beams=5,
                            )
    #torch.cuda.empty_cache()
    result = QA2D_tokenizer.batch_decode(output, skip_special_tokens=True)
    return result

def QA2D_transform(model_path, input_file_path, output_file_path,  batch_size=16):
        QA2D_model = BartForConditionalGeneration.from_pretrained(model_path)
        QA2D_model.cuda()
        with open(output_file_path, mode="w") as writer:
            with open(input_file_path, mode="r")as reader:
                inputs = []
                one_sentence = reader.readline()
                cnt = 0
                while(one_sentence!=""):
                    cnt += 1
                    inputs.append(one_sentence)
                    if (cnt % batch_size == batch_size - 1):
                        outputs = get_response(QA2D_model, inputs)
                        for j in range(0, len(outputs)):
                            writer.write("%s\n" % outputs[j])
                        inputs = []  # clear inputs
                    one_sentence = reader.readline()
                if(len(inputs) != 0): #handle the last (incomplete) batch
                    outputs = get_response(QA2D_model, inputs)
                    for j in range(0, len(outputs)):
                        writer.write("%s\n" % outputs[j])
        writer.close()
