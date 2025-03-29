export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path btc.csv \
  --model_id 'btc_target_close_nontimef_withscale' \
  --model $model_name \
  --data 'custom' \
  --features 'MS' \
  --target 'close' \
  --freq 't' \
  --embed 'timeF' \
  --batch_size 256 \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 8 \
  --top_k 4 \
  --itr 1
