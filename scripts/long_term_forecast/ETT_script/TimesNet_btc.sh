export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path btc.csv \
  --model_id 'btc_96_96_nontimef_no_scale' \
  --model $model_name \
  --data 'custom' \
  --features 'MS' \
  --target 'pct' \
  --freq 't' \
  --embed 'timeF' \
  --batch_size 9600 \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 10 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 8 \
  --d_ff 8 \
  --top_k 4 \
  --itr 1
