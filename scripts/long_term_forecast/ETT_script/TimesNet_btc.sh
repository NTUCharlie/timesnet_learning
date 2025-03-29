export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path btc_nopct.csv \
  --model_id 'btc_96_96_target_close30_nontimef_withscale' \
  --model $model_name \
  --data 'custom' \
  --features 'MS' \
  --target 'close' \
  --freq 't' \
  --embed 'timeF' \
  --batch_size 3600 \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 15 \
  --dec_in 16 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 4 \
  --top_k 4 \
  --itr 1
