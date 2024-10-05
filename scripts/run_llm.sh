#!/bin/bash

PS3='Please enter your choice: '
options=("Run HF Model for text generation"
         "Run Gemini for text generation"
         "Run Gemini for text embedding"
         "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Run HF Model for text generation")
            echo "you chose Run HF Model for text generation."

            export MODEL_NAME=LLaMA-3
            export MODEL_TYPE=meta-llama/Meta-Llama-3-70B

            # export MODEL_NAME=Mistral
            # export MODEL_TYPE=mistralai/Mixtral-8x7B-Instruct-v0.1
                        
            export DATA_PATH=~/QC-LinkPrediction/data/SEMNET/semnet_keywords_updated.txt
            export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features
            export BATCH_SIZE=32
            export MAX_NEW_TOKENS=512
            export EXISTING_DATA_DIR=~/QC-LinkPrediction/data/SEMNET/features/original_semnet_keywords_features # (optional) to re-use already generated features for the same keywords (to save processing time). 
            
            python ~/QC-LinkPrediction/src/llm/text_generation/run_hf_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_path $DATA_PATH \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE \
                --max_new_tokens $MAX_NEW_TOKENS \
                --existing_data_dir $EXISTING_DATA_DIR
 
            break
            ;;
            
        "Run Gemini for text generation")
            echo "you chose Run Gemini for text generation."

            export DATA_PATH=~/QC-LinkPrediction/data/SEMNET/semnet_keywords_updated.txt
            # export OUTPUT_PATH=~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/gemini_keyword_feature.pickle
            # export OUTPUT_PATH=~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/gemini_sum_keyword_feature.pickle
            export OUTPUT_PATH=~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features/gemini_arxiv_qc_sum_keyword_feature.pickle
            export MAX_OUTPUT_TOKENS=512
            export EXISTING_DATA_DIR=~/QC-LinkPrediction/data/SEMNET/features/original_semnet_keywords_features # (optional) to re-use already generated features for the same keywords (to save processing time). 

            python ~/QC-LinkPrediction/src/llm/text_generation/run_gemini.py \
                --data_path $DATA_PATH \
                --output_path $OUTPUT_PATH \
                --max_output_tokens $MAX_OUTPUT_TOKENS \
                --summarization_task
            
            : '
                --existing_data_dir $EXISTING_DATA_DIR
                --summarization_task # used for summarize all LLMs answers
            '
            
            break
            ;;
            
        "Run Gemini for text embedding")
            echo "you chose Run Gemini for text embedding."

            export DATA_DIR=~/QC-LinkPrediction/data/SEMNET/features/semnet_keywords_updated_features
            # export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/semnet_keywords_updated_embeddings
            # export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/arxiv_qc_semnet_keywords_2012_embeddings
            # export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/arxiv_qc_semnet_keywords_2019_embeddings
            # export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/arxiv_qc_semnet_keywords_2021_embeddings
            export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/arxiv_qc_semnet_keywords_2024_embeddings
            # export KEYWORDS_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2012.txt
            # export KEYWORDS_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2019.txt
            # export KEYWORDS_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2021.txt
            export KEYWORDS_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2024.txt
            export TASK_TYPE=document # "query" (default), "document", "semantic_similarity", "classification", "clustering", "question_answering", "fact_verification"
            export EMBED_DIM=768 # 768 (default), 500
            export BATCH_SIZE=128
            
            python ~/QC-LinkPrediction/src/llm/text_embedding/run_gemini.py \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --keywords_path $KEYWORDS_PATH \
                --task_type $TASK_TYPE \
                --embed_dim $EMBED_DIM \
                --batch_size $BATCH_SIZE
            
            : '
                # this is used to match indices in the keyword file.
                # this is used to filter keywords from the existing feature files. e.g., semnet_keywords_updated -> arxiv_qc_semnet_keywords_2017
                --keywords_path $KEYWORDS_PATH \
            '
            
            break
            ;;
        
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done