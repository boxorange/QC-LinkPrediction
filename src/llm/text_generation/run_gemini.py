import pprint
import google.generativeai as genai
import os
import re
import argparse
import pickle
import time
from datetime import timedelta, datetime

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

parser = argparse.ArgumentParser()
    
# general arguments
parser.add_argument('--data_path', action='store')
parser.add_argument('--output_path', action='store')
parser.add_argument('--max_output_tokens', action='store', type=int)
parser.add_argument('--existing_data_dir', action='store', type=str, default='')
parser.add_argument('--summarization_task', action='store_true', default=False)

args = parser.parse_args()

data_path = os.path.expanduser(args.data_path)
output_path = os.path.expanduser(args.output_path)
max_output_tokens = args.max_output_tokens
existing_data_dir = os.path.expanduser(args.existing_data_dir) if args.existing_data_dir != '' else None

## TODO: temporary code for summarization. make this cleaner later. 
summarization_task = args.summarization_task

generation_config = {
    "temperature": 0.9,
    "top_p": 0.9,
    "max_output_tokens": max_output_tokens,
}

# run a model.
st = time.time()


## TODO: temporary code. make this cleaner later. this is to combine LLM features and summarize them. 
if summarization_task:
    '''
    feature_files = [
        "/home/ac.gpark/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/gemini_keyword_feature.pickle",
        "/home/ac.gpark/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/Meta-Llama-3-70B_keyword_feature.pickle",
        "/home/ac.gpark/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/Mixtral-8x7B-Instruct-v0.1_keyword_feature.pickle",
    ]

    keyword_feature_orig = {}
    for file in feature_files:
        with open(file, 'rb') as fin:
            ff = pickle.load(fin)
            
            for k, v in ff.items():
                if k in keyword_feature_orig:
                    keyword_feature_orig[k] += '\n\n' + v
                else:
                    keyword_feature_orig[k] = v

    keyword_feature = {}
    for k, v in keyword_feature_orig.items():
        # remove the prompt in the response if exists.
        prompt = f"What are the features of \"{k}\" in quantum physics?"
        cleaned_feature = v.replace(prompt, '')
        prompt = f"What are the features of {k} in quantum physics?"
        cleaned_feature = cleaned_feature.replace(prompt, '')
        cleaned_feature = cleaned_feature.replace("[closed]", '')
        cleaned_feature = re.sub("\n\s*\n", "\n\n", cleaned_feature)
        keyword_feature[k] = cleaned_feature.strip()
    '''

    ## TODO: temporary code. remove this later. 
    feature_file = "~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/gemini_arxiv_qc_sum_keyword_feature.pickle"
    feature_file = os.path.expanduser(feature_file)
    with open(feature_file, 'rb') as file:
        keyword_feature = pickle.load(file)

else:
    keyword_feature = {}
    with open(data_path) as fin:
        for line in fin.readlines():
            line = line.split('\t')
            keyword = line[1].strip()
            keyword_feature[keyword] = ''

    # load the features if exist.
    if existing_data_dir != None:
        existing_keyword_feature_path = os.path.join(existing_data_dir, 'gemini_keyword_feature.pickle')
        
        with open(existing_keyword_feature_path, 'rb') as fin:
            existing_keyword_feature = pickle.load(fin)

        kw_to_be_added = {}
        for k, v in existing_keyword_feature.items():
            if k in keyword_feature:
                kw_to_be_added[k] = v
                del keyword_feature[k]

        print('>> # of new keywords to be generated:', len(keyword_feature))

# ref: https://ai.google.dev/docs/safety_setting_gemini
safety_settings=[
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    }
]


"""
models/gemini-1.0-pro

Rate limits
Free:
    15 RPM (RPM: Requests per minute)
    32,000 TPM (TPM: Tokens per minute)
    1,500 RPD (RPD: Requests per day)

ref: https://ai.google.dev/gemini-api/docs/models/gemini

"""

for idx, (keyword, feature) in enumerate(keyword_feature.items()):
    
    
    
    ## TODO: temporary code. make this cleaner later. 
    if summarization_task:
        if feature['is_summarized'] == True:
            continue
    
    
    
    # ref: finish reason - https://ai.google.dev/api/python/google/ai/generativelanguage/Candidate/FinishReason
    while True:
        
        ## TODO: temporary code. make this cleaner later. 
        if summarization_task:
            prompt = f"Summarize this text about the features of \"{keyword}\". Text: {feature['text']}"
        else:
            prompt = f"What are the features of \"{keyword}\" in quantum physics?"

        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        
        finish_reason = response.candidates[0].finish_reason
        
        # The passages "recited" from copyrighted material in the foundational LLM's training data are not allowed to be generated.
        # ref: https://github.com/googleapis/google-cloud-python/blob/main/packages/google-ai-generativelanguage/google/ai/generativelanguage_v1beta/types/generative_service.py#L438-L464
        '''
        Defines the reason why the model stopped generating tokens.

        Values:
            FINISH_REASON_UNSPECIFIED (0):
                Default value. This value is unused.
            STOP (1):
                Natural stop point of the model or provided stop sequence.
            MAX_TOKENS (2):
                The maximum number of tokens as specified in the request was reached.
            SAFETY (3):
                The candidate content was flagged for safety reasons.
            RECITATION (4):
                The candidate content was flagged for recitation reasons.
            OTHER (5):
                Unknown reason.
        
        FINISH_REASON_UNSPECIFIED = 0
        STOP = 1
        MAX_TOKENS = 2
        SAFETY = 3
        RECITATION = 4
        OTHER = 5
        '''
        if finish_reason == 1:
            # debug
            print(">> response.text:", response.text)
            print(">> Number of processed keywords:", idx + 1)
            print('======================================================================')
            
            ## TODO: temporary code. make this cleaner later. 
            if summarization_task:
                keyword_feature[keyword]['summary'] = response.text
                keyword_feature[keyword]['is_summarized'] = True
            else:
                keyword_feature[keyword] = response.text
            
            # Due to rate limits and unexpected errors, save the results in real-time.
            with open(output_path, 'wb') as fout:
                pickle.dump(keyword_feature, fout, protocol=pickle.HIGHEST_PROTOCOL)

            # Rate limit: 60 requests per minute
            # ref: https://ai.google.dev/models/gemini
            time.sleep(2)
            
            break


# add the existing keyword features.
if existing_data_dir != None:
    keyword_feature.update(kw_to_be_added)
    keyword_feature = dict(sorted(keyword_feature.items()))

print('>> total number of keywords:', len(keyword_feature))

# Due to rate limits and unexpected errors, save the results in real-time.
'''
with open(output_path, 'wb') as fout:
    pickle.dump(keyword_feature, fout, protocol=pickle.HIGHEST_PROTOCOL)
'''

et = time.time()
elapsed_time = et - st
exec_time = timedelta(seconds=elapsed_time)
exec_time = str(exec_time)
print('>> Execution time in hh:mm:ss:', exec_time)
