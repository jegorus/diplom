from enum import Enum

class Config:
  BASE_MODEL = "roberta-base"

  class Dataset:
    MRPC = ['glue', 'mrpc']

  class Model(Enum):
    FULLFT = 1
    LORA = 2