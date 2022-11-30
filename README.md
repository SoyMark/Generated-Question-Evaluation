# How to run the program

**Note that this doc only describe how to run the program. My survey, methods and experiments are explained in `Report.pdf`**



#### Step1 Installation and Setup

Download models

```bash
# download the span extractor model:
python -m paq.download -v -n models.answer_extractors.answer_extractor_nq_base
# download the qgen model:
python -m paq.download -v -n models.qgen.qgen_multi_base
# download GEC model
python gector/download.py
mv bert_0_gectorv2.th gec_model
```

Download required packages

```bash
conda create -n QE python=3.7
conda activate QE
#install pytorch 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# For Spacy:
conda install -c conda-forge spacy
conda install -c conda-forge cupy
pip install en_core_web_sm-2.1.0.tar.gz
pip install -r requirements.txt
```



#### Step 2 Generate QA-pairs and evaluate

First, extract possible answers from the passage.

```bash
# First, extract possible answers from the passage.
python -m paq.generation.answer_extractor.extract_answers \
    --passages_to_extract_from my_passages_with_titles.jsonl \
    --output_path my_passages_with_answers.jsonl \
    --path_to_config generator_configs/answer_extractor_configs/learnt_answer_extractor_config.json \
    --verbose

# Then, generate questions from the passage and answers.
python -m paq.generation.question_generator.generate_questions \
    --passage_answer_pairs_to_generate_from my_passages_with_answers.jsonl \
    --output_path my_generated_questions.jsonl \
    --path_to_config generator_configs/question_generator_configs/question_generation_config.json \
    --verbose
# Finally, run main.py to evaluate and rank the questions.
python main.py
```

Sorted questions are listed in `final_result.txt` with their scores.



If you want to try other stories, just modify the 'passage' and 'title' value in `my_passages_with_titles.jsonl`. Besides, some byproducts are produced during the ranking procedure. Question corrected by GEC model are listed in `gec_output/output.txt`. Declarative answer sentences generated from QA-pairs are listed in `QA2D/output.txt`. You can check them if you like.