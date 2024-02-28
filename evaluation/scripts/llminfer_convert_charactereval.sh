CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --max_new_tokens 2048 \
    --cutoff_len 2048 \
    --model_name_or_path chatglm3-6b \
    --dataset characterEval_in_sharegpt_fakelabel \
    --template qwen \
    --output_dir results/chatglm3-6b/characterEval \
    --per_device_eval_batch_size 1 \
    --max_samples 5000 \
    --predict_with_generate \
    --fp16 \


python convert_characterEval_format.py ./results/chatglm3-6b/characterEval/generated_predictions.jsonl ../CharacterEval/data/id2metric.jsonl ./data/characterEval_in_sharegpt_fakelabel.jsonl  ../CharacterEval/data/test_data.jsonl ./results/chatglm3-6b/characterEval/generated_trans_chatglm3-6b_ragged_charactereval.jsonl
