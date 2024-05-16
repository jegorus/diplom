from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, PeftModel, IA3Config, AdaLoraConfig
import numpy as np
from datasets import load_metric
import time
from collections.abc import Iterable


from config import Config
from utils.custom_trainer import CustomTrainer
from utils.utils import print_trainable_named_parameters
from data import DatasetHandler
from convert import Converter



class ModelHandler:

  def __init__(self, dataset_name, ft_type):
    self.dataset_handler = DatasetHandler(dataset_name, ft_type)
    self.dataset_name = dataset_name
    self.finetuning_type = ft_type
    self.metric = load_metric(*self.dataset_handler.dataset_name)
  
  def reset_model(self, num_labels=2, saved_model_path=None):
    path = saved_model_path if saved_model_path else Config.BASE_MODEL
    self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)

    
  def run_experiment(self, training_args, adapter_config=None, seed=42, saved_model_path=None, print_params=False):
    set_seed(seed)

    if  self.dataset_name == Config.Dataset.CB:
      self.reset_model(3, saved_model_path)
    else:
      self.reset_model(2, saved_model_path)

    if self.finetuning_type == Config.Model.LORA or \
       self.finetuning_type == Config.Model.IA3 or \
       self.finetunint_type == Config.Model.ADALORA:
       if adapter_config is None:
        raise ValueError("finetuning type uses adapter, but 'adapter_config' is None")

       if saved_model_path:
          self.model = PeftModel.from_pretrained(self.model, saved_model_path, is_trainable=True)
 
       else:

        if self.finetuning_type == Config.Model.LORA:
          self.model = get_peft_model(self.model, LoraConfig(**adapter_config))
        elif self.finetuning_type == Config.Model.IA3:
          self.model = get_peft_model(self.model, IA3Config(**adapter_config))
        elif self.finetuning_type == Config.Model.ADALORA:
          self.model = get_peft_model(self.model, AdaLoraConfig(**adapter_config))

    # if self.finetuning_type == Config.Model.FULLFT:
    #    if lora_config is not None:
    #      raise ValueError("finetuning type is 'FULLFT', but 'lora_config' provided")

    if print_params == True:
      print_trainable_named_parameters(self.model)

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


  def run_experiments_and_convert(self, training_args, seeds, convert_name, convert_nums, adapter_config=None, saved_model_path=None, print_params=False):
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
      trainer = self.run_experiment(training_args, adapter_config, seed, saved_model_path, print_params)
      Converter.log_to_csv(trainer.state.log_history, convert_name, convert_num)
      # df = Converter.log_to_pandas(trainer.state.log_history)
      # df = Converter.prettify_results(df)
      # Converter.pandas_to_csv_name(df, "exp-" + convert_name + "-" + str(convert_num) + "-prettified")
      print(str(convert_num) + " converted")
    
    return trainer
    


def compute_metrics_inner(eval_preds, metric):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


def compute_metrics(metric):
  def wrapper(eval_preds):
    return compute_metrics_inner(eval_preds, metric)
  return wrapper

