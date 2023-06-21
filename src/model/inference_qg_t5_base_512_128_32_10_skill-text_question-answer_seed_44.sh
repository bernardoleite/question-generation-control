#!/usr/bin/env bash

for ((i=44; i <= 44; i++))
do
	taskset --cpu-list 1-24 python train.py \
	 --checkpoint_model_path "../../checkpoints/qg_t5_base_512_128_32_10_skill-text_question-answer_seed_44/model-epoch=XX-val_loss=YY.ckpt" \
	 --predictions_save_path "../../predictions/qg_t5_base_512_128_32_10_skill-text_question-answer_seed_44/model-epoch=XX-val_loss=YY/" \
	 --test_path "../../data/FairytaleQA_Dataset/processed_ctrl_sk_a/test.json" \
	 --model_name "t5-base" \
	 --tokenizer_name "t5-base" \
	 --batch_size 32 \
	 --max_len_input 512 \
	 --max_len_output 128 \
	 --encoder_info "skill_text" \
	 --decoder_info "question_answer" \
	 --num_beams 5 \
	 --num_return_sequences 1 \
	 --repetition_penalty 1 \
	 --length_penalty 1 \
	 --seed_value ${i}
done