from simpletransformers.classification import ClassificationModel
import os
os.environ["HF_HOME"] = "/disk/public_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/disk/public_data/huggingface/hub"
train_args = {
    'learning_rate': 1e-5,
    'max_seq_length': 512,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 4,
    'num_train_epochs': 4,
    'gradient_accumulation_steps': 2,
    'no_cache': True,
    'use_cached_eval_features': False,
    'save_model_every_epoch': False,
    'output_dir': "bart-squadv2",
    'eval_batch_size': 8,
    'fp16_opt_level': 'O2',
    }
model = ClassificationModel('roberta', 'a-ware/roberta-large-squadv2', num_labels=2, args=train_args)

predictions, raw_outputs = model.predict([["my dog is an year old. he loves to go into the rain"]])
print("predictions",predictions)
print("raw_outputs",raw_outputs)