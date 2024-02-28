CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --max_new_tokens 2048 \
    --cutoff_len 2048 \
    --model_name_or_path Haruhi-Zero-14B-0_5 \
    --dataset charactereval \
    --template qwen \
    --output_dir results/Haruhi-Zero-14B-0_5/characterEval \
    --per_device_eval_batch_size 1 \
    --max_samples 5000 \
    --predict_with_generate \
    --fp16 \



