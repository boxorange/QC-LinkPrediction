#!/bin/bash

PS3='Please enter your choice: '
options=("Create Quantum Computing SEMNET keywords."
         "Create datasets."
         "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Create Quantum Computing SEMNET keywords.")
            echo "you chose Create Quantum Computing SEMNET keywords."

            export DATA_DIR=~/QC-LinkPrediction/data/SEMNET
            export INPUT_FILE=SEMNET_concepts_updated.csv
            export OUTPUT_FILE=qc_semnet_keywords_updated.txt

            python ~/QC-LinkPrediction/src/utils/qc_semnet_keywords_creator.py \
                --data_dir $DATA_DIR \
                --input_file $INPUT_FILE \
                --output_file $OUTPUT_FILE

            break
            ;;
            
        "Create datasets.")
            echo "you chose Create datasets."

            export CORPUS_FILE_PATH=~/QC-LinkPrediction/data/arxiv-quant-ph-06-15-2024
            export ORIG_SEMNET_CONCEPT_FILE_PATH=~/QC-LinkPrediction/data/SEMNET/qc_semnet_keywords_updated.txt
            export ARXIV_SEMNET_CONCEPT_FILE_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2024.txt
            export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET
            export TRAIN_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv
            export VALID_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2022_and_2022.tsv
            export VALID_TEST_RATIO=0.2857 # 1/7 = 0.1429, 2/7 = 0.2857 (7:1:2 -> train/valid/test ratio), 0.125 (8:1:1 -> train/valid/test ratio)
            export START_YEAR=2007
            export END_YEAR=2021

            python ~/QC-LinkPrediction/src/utils/arxiv_semnet_generator.py \
                --corpus_file_path $CORPUS_FILE_PATH \
                --orig_semnet_concept_file_path $ORIG_SEMNET_CONCEPT_FILE_PATH \
                --arxiv_semnet_concept_file_path $ARXIV_SEMNET_CONCEPT_FILE_PATH \
                --output_dir $OUTPUT_DIR \
                --train_data_path $TRAIN_DATA_PATH \
                --valid_data_path $VALID_DATA_PATH \
                --valid_test_ratio $VALID_TEST_RATIO \
                --start_year $START_YEAR \
                --end_year $END_YEAR

            break
            ;;
        
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done