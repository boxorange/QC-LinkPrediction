#!/bin/bash
PS3='Please enter your choice: '
options=("Generate Node2Vec Embedding"
         "Run NCN"
         "Run GNN"
         "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Generate Node2Vec Embedding")
            echo "you chose Generate Node2Vec Embedding."
            
            export DATA_DIR=~/QC-LinkPrediction/data/SEMNET/train_valid_test
            export OUTPUT_DIR=~/QC-LinkPrediction/data/SEMNET/embeds/arxiv_qc_semnet_keywords_2024_embeddings

            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/get_n2v_embedding.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --embedding_dim 768 \
                --device 0

            break
            ;;
            
        "Run NCN")
            echo "you chose Run NCN."
            
            export PROCESSED_DIR=~/QC-LinkPrediction/data/SEMNET/NCN/processed
            export OUTPUT_DIR=~/QC-LinkPrediction/results/NCN

            export TRAIN_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv
            export VALID_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2022_and_2022.tsv
            export TEST_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2023_and_2024.tsv
            export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/llm_blender_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_td_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_td_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_td_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/llm_blender_td_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/max_pooled_keyword_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/mean_pooled_keyword_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_arxiv_qc_sum_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/n2v_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/deepwalk_embedding.tsv
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/line_embedding.tsv

            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_ncn_CoraCiteseerPubmed.py \
                --dataset semnet \
                --processed_dir $PROCESSED_DIR \
                --output_dir $OUTPUT_DIR \
                --train_data_path $TRAIN_DATA_PATH \
                --valid_data_path $VALID_DATA_PATH \
                --test_data_path $TEST_DATA_PATH \
                --embed_path $EMBED_PATH \
                --predictor cn1 \
                --gnnlr 0.001 \
                --prelr 0.001 \
                --l2 1e-7 \
                --predp 0.1 \
                --gnndp 0.1 \
                --gnnedp 0.0 \
                --preedp 0.4 \
                --xdp 0.7 \
                --tdp 0.3 \
                --mplayers 2 \
                --nnlayers 2 \
                --hiddim 768 \
                --epochs 100 \
                --eval_steps 5 \
                --runs 10 \
                --kill_cnt 10 \
                --batch_size 4096 \
                --testbs 8192 \
                --pt 0.75 \
                --probscale 4.5 \
                --proboffset 1.5 \
                --alpha 1.0 \
                --ln \
                --lnnn \
                --model puregcn \
                --maskinput \
                --jk \
                --use_xlin \
                --tailact \
                --metric AUC \
                --loadx \
                --save_log \
                --device 0

            break
            ;;
        
        "Run GNN")
            echo "you chose Run GNN."
            
            export DATA_DIR=~/QC-LinkPrediction/data/SEMNET/train_valid_test
            export OUTPUT_DIR=~/QC-LinkPrediction/results/GNN
            export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/gemini_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/Meta-Llama-3-70B_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/Mixtral-8x7B-Instruct-v0.1_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/llm_blender_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/gemini_td_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/Meta-Llama-3-70B_td_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/Mixtral-8x7B-Instruct-v0.1_td_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/llm_blender_td_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/max_pooled_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/mean_pooled_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/gemini_arxiv_qc_sum_embedding
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/n2v_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/deepwalk_gnn_feature
            # export EMBED_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/pt/line_gnn_feature
            
            # used for BUDDY
            export PROCESSED_DIR=~/QC-LinkPrediction/data/SEMNET/BUDDY/processed
            export TRAIN_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv
            export VALID_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2022_and_2022.tsv
            export TEST_DATA_PATH=~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2023_and_2024.tsv
            export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/llm_blender_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_td_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_td_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_td_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/llm_blender_td_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/max_pooled_keyword_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/mean_pooled_keyword_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/n2v_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/deepwalk_embedding.tsv
            # export EMBED_TSV_PATH=~/QC-LinkPrediction/data/SEMNET/embeds/tsv/line_embedding.tsv

            ## buddy
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_buddy_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --processed_dir $PROCESSED_DIR \
                --train_data_path $TRAIN_DATA_PATH \
                --valid_data_path $VALID_DATA_PATH \
                --test_data_path $TEST_DATA_PATH \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --embed_tsv_path $EMBED_TSV_PATH \
                --model BUDDY \
                --lr 0.01 \
                --label_dropout 0.1 \
                --feature_dropout 0.1 \
                --l2 1e-4 \
                --hidden_channels 256 \
                --epochs 9999 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --load_features \
                --batch_size 1024 \
                --save_log \
                --device 0
            
            
            : '
            ## mlp
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_gnn_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --gnn_model mlp_model \
                --lr 0.01 \
                --dropout 0.3 \
                --l2 1e-7 \
                --num_layers 1 \
                --num_layers_predictor 3 \
                --hidden_channels 256 \
                --epochs 100 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --batch_size 1024 \
                --save_log \
                --device 0
                
            ## sage
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_gnn_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --gnn_model SAGE \
                --lr 0.01 \
                --dropout 0.5 \
                --l2 1e-7 \
                --num_layers 1 \
                --num_layers_predictor 3 \
                --hidden_channels 128 \
                --epochs 100 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --batch_size 1024 \
                --save_log \
                --device 0

            ## gcn
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_gnn_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --gnn_model GCN \
                --lr 0.007 \
                --dropout 0.3 \
                --l2 0 \
                --num_layers 1 \
                --num_layers_predictor 3 \
                --hidden_channels 256 \
                --epochs 100 \
                --runs 10 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --batch_size 1024 \
                --save_log \
                --device 0

            ## gae
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_gae_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --gnn_model GCN \
                --lr 0.01 \
                --dropout 0.1 \
                --l2 1e-7 \ # 1e-4 (CiteSeer), 0 (PubMed)
                --num_layers 1 \
                --num_layers_predictor 2 \
                --hidden_channels 256 \
                --epochs 9999 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --batch_size 1024 \
                --save_log \
                --device 0
            
            ## buddy
            python ~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking/exist_setting_small/main_buddy_CoraCiteseerPubmed.py \
                --data_name semnet \
                --data_dir $DATA_DIR \
                --processed_dir $PROCESSED_DIR \
                --train_data_path $TRAIN_DATA_PATH \
                --valid_data_path $VALID_DATA_PATH \
                --test_data_path $TEST_DATA_PATH \
                --output_dir $OUTPUT_DIR \
                --embed_path $EMBED_PATH \
                --embed_tsv_path $EMBED_TSV_PATH \
                --model BUDDY \
                --lr 0.01 \
                --label_dropout 0.1 \
                --feature_dropout 0.1 \
                --l2 1e-4 \
                --hidden_channels 256 \
                --epochs 9999 \
                --kill_cnt 10 \
                --eval_steps 5 \
                --load_features \
                --batch_size 1024 \
                --save_log \
                --device 0

            '

            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done