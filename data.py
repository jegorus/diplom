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
      self.tokenized_dataset = self.raw_dataset.map(self.tokenize_function, batched=True)
      if self.finetuning_type == Config.Model.LORA:
        self.tokenized_dataset["train"].rename_column("label", "labels")
        self.tokenized_dataset["validation"].rename_column("label", "labels")

  def tokenize_function(cls, examples):
    return(cls.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True))