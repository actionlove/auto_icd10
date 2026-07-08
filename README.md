# auto_icd10: an automated pipeline to predict ICD-10 codes from clinical transcripts
This repo contains codes and running guidelines of an automated pipeline `auto_icd10`, which predicts ICD-10 codes from doctor/patient conversation raw transcript end-to-end.

To adderss the challenges of noisy transcripts, hierarchical ICD-10 codes and LLM hallucination, a 3-stage method is proposed in `auto_icd10`:
1. Extract: Key, unambiguious diagnoses are extracted from input transcript and dumped into JSON formatted SOAP note and problem list;
2. Retrieval: ICD-10 codes prediction from the official code table (never let the LLM invent code strings);
3. Verify: verify each candidate against the evidence and select final codes with confidence + evidence spans.

For step 1 and 3, we leverage propietary LLMs.

For step 2, we propose RAG based method to retrieve-then-rank the top-K ICD-10 codes using extracted problem list as query.


Here are the achieved milestones in this repo:

## Milestone 1: Implement prediction pipeline v1 using GPT in step 1 and 3, and BM25 retrieval in step 2
