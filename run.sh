python albert_ner.py --task_name ner --do_train true --do_eval true --data_dir data --vocab_file ./albert_config/vocab.txt --bert_config_file ./albert_base_zh/albert_config_base.json --max_seq_length 128 --train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 3 --output_dir albert_base_ner_checkpoints

