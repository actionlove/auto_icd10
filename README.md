# auto_icd10: an automated pipeline to predict ICD-10 codes from clinical transcripts
This repo contains codes and running guidelines of an automated pipeline `auto_icd10`, which predicts ICD-10 codes from doctor/patient conversation raw transcript end-to-end.

To adderss the challenges of noisy transcripts, hierarchical ICD-10 codes and LLM hallucination, a 3-stage method is proposed in `auto_icd10`:
1. Extract: Key, unambiguious diagnoses are extracted from input transcript and dumped into JSON formatted SOAP note and problem list;
2. Retrieval: ICD-10 codes prediction from the official code table (never let the LLM invent code strings);
3. Verify: verify each candidate against the evidence and select final codes with confidence + evidence spans.

For step 1 and 3, we leverage propietary LLMs.

For step 2, we propose RAG based method to retrieve-then-rank the top-K ICD-10 codes using extracted problem list as query.


# Important milestones:

## Milestone 1: Implement prediction pipeline v1 using GPT in step 1 and 3, and BM25 retrieval in step 2

## Milestone 2: Implement prediction pipeline v2 with confidence estimation


# How to run this repo locally?
1. Run setup script for .venv and dependencies
```
$ git clone https://github.com/actionlove/auto_icd10
$ cd auto_icd10
$ source setup.sh
```

2. Under root of the repo, add .env file with API KEY
```
$ echo "OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini" > .env
```

3. Run unitest to confirm setup correctly
```
$ source run_tests.sh
$ source run_api_tests.sh
```

4. Run demo locally
```
$ python app.py
```
Once you see message:
```
* Running on local URL:  http://127.0.0.1:7860
```
Open web browser with URL http://127.0.0.1:7860 to interact with the demo.
