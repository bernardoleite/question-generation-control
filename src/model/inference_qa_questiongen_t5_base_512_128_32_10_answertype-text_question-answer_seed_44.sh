#!/usr/bin/env bash

for ((i=44; i <= 44; i++))
do
	taskset --cpu-list 1-24 python train.py \
	 --checkpoint_model_path "../../checkpoints/qa_t5_base_512_128_32_10_question-text_answer_seed_44/model-epoch=XX-val_loss=YY.ckpt" \
	 --predictions_save_path "../../predictions/qa_questiongen_t5_base_512_128_32_10_answertype-text_question-answer_seed_44/" \
	 --test_path "../../predictions/qg_t5_base_512_128_32_10_answer-text_question_seed_44/model-epoch=KK-val_loss=ZZ/predictions.json" \
	 --model_name "t5-base" \
	 --tokenizer_name "t5-base" \
	 --batch_size 32 \
	 --max_len_input 512 \
	 --max_len_output 128 \
	 --encoder_info "questiongen_text" \
	 --decoder_info "answer" \
	 --num_beams 5 \
	 --num_return_sequences 1 \
	 --repetition_penalty 1 \
	 --length_penalty 1 \
	 --seed_value ${i}
done