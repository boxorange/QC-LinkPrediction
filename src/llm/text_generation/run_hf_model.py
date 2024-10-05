import os
import time
import string
import argparse
import pickle
import itertools
import warnings
import torch, gc
from datetime import timedelta, datetime

# Change the HF cache directory to a scratch directory. 
# Make sure to set the variable before importing transformers module (including indirect import through galai).
# ref: https://github.com/paperswithcode/galai/blob/main/notebooks/Introduction%20to%20Galactica%20Models.ipynb
# os.environ["TRANSFORMERS_CACHE"] = "/scratch/ac.gpark/.cache/huggingface"
os.environ["HF_HOME"] = "/scratch/ac.gpark/.cache/huggingface"

# ref: https://huggingface.co/docs/transformers/v4.21.1/en/troubleshooting#troubleshoot
#os.environ["CUDA_VISIBLE_DEVICES"] = "" # to run on CPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # to get a better traceback from the GPU error


from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    GenerationConfig, 
    LlamaForCausalLM, 
    LlamaTokenizer, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig,
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

def get_response(model, generation_config, tokenizer, batch_input_texts):
    inputs = tokenizer(batch_input_texts, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    generated_sequence = model.generate(input_ids=input_ids, generation_config=generation_config)
    generated_text = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_text
    

def load_model(
    model_name,
    model_type, 
    max_new_tokens, 
    device_map,
):
    if model_name == 'LLaMA-3':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left" # "right", left padding performs better than right padding and allow batched inference.
        tokenizer.truncation_side = "left" 

        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = 1.0 # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            device_map=device_map
        )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(model.config.vocab_size + 1)
    
    elif model_name == 'Mistral':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.unk_token_id
        
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.pad_token_id = tokenizer.unk_token_id
        generation_config.max_new_tokens = max_new_tokens
        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            device_map=device_map
        )
        
    else:
        raise ValueError("Invalid model name: " + model_name)	

    return model, tokenizer, generation_config


def main():
    parser = argparse.ArgumentParser()
    
    # general arguments
    parser.add_argument('--model_name', action='store')
    parser.add_argument('--model_type', action='store')
    parser.add_argument('--data_path', action='store')
    parser.add_argument('--output_dir', action='store')
    parser.add_argument('--batch_size', action='store', type=int, default=1)
    parser.add_argument('--max_new_tokens', action='store', type=int, default=200)
    parser.add_argument('--existing_data_dir', action='store', type=str, default='')

    args = parser.parse_args()
    
    model_name = args.model_name
    model_type = args.model_type
    data_path = os.path.expanduser(args.data_path)
    output_dir = os.path.expanduser(args.output_dir)
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    existing_data_dir = os.path.expanduser(args.existing_data_dir) if args.existing_data_dir != '' else None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device_map = "auto"

    if model_name == 'LLaMA-3':
        model_max_len = 8192
    elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
        model_max_len = 32768 # 8192 * 4

    # Load a model, a tokenizer, and configure generation settings.
    model, tokenizer, generation_config = load_model(
                                            model_name,
                                            model_type, 
                                            max_new_tokens, 
                                            device_map,
                                        )

    # run a model.
    st = time.time()
    
    keyword_feature = {}
    with open(data_path) as fin:
        for line in fin.readlines():
            line = line.split('\t')
            keyword = line[1].strip()
            keyword_feature[keyword] = ''

    # load the features if exist.
    if existing_data_dir != None:
        existing_keyword_feature_path = os.path.join(existing_data_dir, model_type.rsplit('/', 1)[1] + '_keyword_feature.pickle')
        
        with open(existing_keyword_feature_path, 'rb') as fin:
            existing_keyword_feature = pickle.load(fin)

        kw_to_be_added = {}
        for k, v in existing_keyword_feature.items():
            if k in keyword_feature:
                kw_to_be_added[k] = v
                del keyword_feature[k]

        print('>> # of new keywords to be generated:', len(keyword_feature))

    start = 0
    stop = start + batch_size
    
    while True:
        batch_data = itertools.islice(keyword_feature.keys(), start, stop)
        batch_data = list(batch_data)

        batch_input_texts = []
        for item in batch_data:
            prompt = f"What are the features of \"{item}\" in quantum physics?"
            batch_input_texts.append(prompt)
        
        results = get_response(model, generation_config, tokenizer, batch_input_texts)

        assert len(batch_data) == len(results)
        
        for keyword, feature in zip(batch_data, results):
            keyword_feature[keyword] = feature
        
        print(f">> batch processed - len(keyword_feature): {len(keyword_feature)}, start: {start}, stop: {stop}")

        if stop >= len(keyword_feature):
            break

        start = stop
        stop = start + batch_size

    # add the existing keyword features.
    if existing_data_dir != None:
        keyword_feature.update(kw_to_be_added)
        keyword_feature = dict(sorted(keyword_feature.items()))
    
    print('>> total number of keywords:', len(keyword_feature))
    
    output_path = os.path.join(output_dir, model_type.rsplit('/', 1)[1] + '_keyword_feature.pickle')
    
    with open(output_path, 'wb') as fout:
        pickle.dump(keyword_feature, fout, protocol=pickle.HIGHEST_PROTOCOL)

    et = time.time()
    elapsed_time = et - st
    exec_time = timedelta(seconds=elapsed_time)
    exec_time = str(exec_time)
    print('>> Execution time in hh:mm:ss:', exec_time)
    

if __name__ == "__main__":
    main()