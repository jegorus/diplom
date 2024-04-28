from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
import numpy as np
from datasets import load_metric
import time
from collections.abc import Iterable


from config import Config
from utils.custom_trainer import CustomTrainer
from data import DatasetHandler
from convert import Converter



class ModelHandler:

  def __init__(self, dataset_name, ft_type):
    self.dataset_handler = DatasetHandler(dataset_name, ft_type)
    self.finetuning_type = ft_type
    self.metric = load_metric(*self.dataset_handler.dataset_name)
  
  def reset_model(self):
    self.model = AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL, num_labels=2)

    
  def run_experiment(self, training_args, lora_config=None, seed=42):
    set_seed(seed)

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

    trainer.set_my_custom_start_time(time.time())
    trainer.train()
    return trainer


  def run_experiments_and_convert(self, training_args, seeds, convert_name, convert_nums, lora_config=None):
    if isinstance(seeds, int):
      seeds = [seeds]
    if isinstance(convert_nums, int):
      convert_nums = [convert_nums]

    if not isinstance(seeds, Iterable):
      raise ValueError("seeds should be int or iterable")
    if not isinstance(convert_nums, Iterable):
      raise ValueError("convert_nums should be int or iterable")

    if not len(seeds) == len(convert_nums):
      raise ValueError("seeds and convert_numds should have the same length")

    for seed, convert_num in zip(seeds, convert_nums):
      trainer = self.run_experiment(training_args, lora_config, seed)
      Converter.log_to_csv(trainer.state.log_history, convert_name, convert_num)
      df = Converter.log_to_pandas(trainer.state.log_history)
      df = Converter.prettify_results(df)
      Converter.pandas_to_csv_name(df, "exp-" + convert_name + "-" + str(convert_num) + "-prettified")
      print(str(convert_num) + " converted")
    


def compute_metrics_inner(eval_preds, metric):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


def compute_metrics(metric):
  def wrapper(eval_preds):
    return compute_metrics_inner(eval_preds, metric)
  return wrapper

