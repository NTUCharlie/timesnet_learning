export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet_norevin

python -u run.py \
  --task_name long_term_forecast_no_revin \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path btc.csv \
  --model_id 'btc_pct30_norevin' \
  --model $model_name \
  --data 'custom' \
  --features 'MS' \
  --target 'pct' \
  --freq 't' \
  --embed 'timeF' \
  --batch_size 2048 \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 8 \
  --top_k 4 \
  --num_kernels 6 \
  --itr 1 \
  --learning_rate 0.001 \
  --lradj constant \
  --train_epochs 100
