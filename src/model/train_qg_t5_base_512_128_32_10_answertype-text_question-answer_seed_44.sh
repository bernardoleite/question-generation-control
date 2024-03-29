#!/usr/bin/env bash

for ((i=44; i <= 44; i++))
do
	taskset --cpu-list 1-24 python train.py \
	 --dir_model_name "qg_t5_base_512_128_8_10_answertype-text_question-answer_seed_${i}" \
	 --model_name "t5-base" \
	 --tokenizer_name "t5-base" \
	 --train_path "../../data/FairytaleQA_Dataset/processed_ctrl_sk_a/train.json" \
	 --val_path "../../data/FairytaleQA_Dataset/processed_ctrl_sk_a/val.json" \
	 --test_path "../../data/FairytaleQA_Dataset/processed_ctrl_sk_a/test.json" \
	 --max_len_input 512 \
	 --max_len_output 128 \
	 --encoder_info "answertype_text" \
	 --decoder_info "question_answer" \
	 --batch_size 32 \
	 --max_epochs 10 \
	 --patience 2 \
	 --optimizer "AdamW" \
	 --learning_rate 0.0001 \
	 --epsilon 0.000001 \
	 --num_gpus 1 \
	 --seed_value ${i}
done