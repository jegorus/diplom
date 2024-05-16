from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import Config


class DatasetHandler:
  tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
  data_collator = DataCollatorWithPadding(tokenizer)
  def __init__(self, name, ft_type):
      self.dataset_name = name
      self.finetuning_type = ft_type
      self.raw_dataset = load_dataset(*self.dataset_name)
      self.tokenized_dataset = self.raw_dataset.map(self.tokenize_function_handler(self.dataset_name), batched=True)

      # if self.finetuning_type == Config.Model.LORA:
      #   self.tokenized_dataset["train"].rename_column("label", "labels")
      #   self.tokenized_dataset["validation"].rename_column("label", "labels")

  def tokenize_function_handler(self, dataset_name):

    if dataset_name == Config.Dataset.MRPC or \
    dataset_name == Config.Dataset.RTE or \
    dataset_name == Config.Dataset.WNLI:
      def tokenize_func_mrpc(examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
      return tokenize_func_mrpc

    elif dataset_name == Config.Dataset.WSC_FIXED:
      def tokenize_func_wsc(examples):
        return self.tokenizer(examples["text"], [name + "; " + pronoun for name, pronoun in zip(examples["span1_text"], examples["span2_text"])], truncation=True)
      return tokenize_func_wsc

    elif dataset_name == Config.Dataset.SST2 or \
         dataset_name == Config.Dataset.COLA:
      def tokenize_func_sst(examples):
        return self.tokenizer(examples["sentence"], truncation=True)
      return tokenize_func_sst

    elif dataset_name == Config.Dataset.QNLI:
      def tokenize_func_qnli(examples):
        return self.tokenizer(examples["question"], examples["sentence"], truncation=True)
      return tokenize_func_qnli

    elif dataset_name == Config.Dataset.CB:
      def tokenize_func_cb(examples):
        return self.tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
      return tokenize_func_cb
    
    elif dataset_name == Config.Dataset.BOOLQ:
      def tokenize_func_boolq(examples):
        return self.tokenizer(examples["question"], examples["passage"], truncation=True)
      return tokenize_func_boolq

    

    else: 
      raise ValueError("wrong dataset name")
