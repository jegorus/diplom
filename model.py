from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from datasets import load_metric
import time

from config import Config
from utils.custom_trainer import CustomTrainer
from data import DatasetHandler


class ModelHandler:

  def __init__(self, dataset_name, ft_type):
    self.dataset_handler = DatasetHandler(dataset_name, ft_type)
    self.finetuning_type = ft_type
    self.reset_model()
    self.metric = load_metric(*self.dataset_handler.dataset_name)
  
  def reset_model(self):
    self.model = AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL, num_labels=2)

    
  def run_experiment(self, training_args, lora_config=None):
    self.reset_model()

    if self.finetuning_type == Config.Model.LORA:
      if lora_config is None:
        raise ValueError("finetuning type is 'LORA', but 'lora_config' is None")
      self.model = get_peft_model(self.model, LoraConfig(**lora_config))

    if self.finetuning_type == Config.Model.FULLFT:
        if lora_config is not None:
          raise ValueError("finetuning type is 'FULLFT', but 'lora_config' provided")

    trainer = CustomTrainer(
    self.model,
    TrainingArguments(**training_args),
    train_dataset = self.dataset_handler.tokenized_dataset["train"],
    eval_dataset = self.dataset_handler.tokenized_dataset["validation"],
    data_collator = self.dataset_handler.data_collator,
    tokenizer = self.dataset_handler.tokenizer,
    compute_metrics=compute_metrics(self.metric),
    )

    my_custom_start_time = time.time()
    trainer.train()


def compute_metrics_inner(eval_preds, metric):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


def compute_metrics(metric):
  def wrapper(eval_preds):
    return compute_metrics_inner(eval_preds, metric)
  return wrapper

