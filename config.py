from enum import Enum

class Config:
  BASE_MODEL = "roberta-base"

  class Dataset:
    MRPC = ['glue', 'mrpc']
    WSC_FIXED = ['super_glue', 'wsc.fixed']
    SST2 = ['glue', 'sst2']
    QNLI = ['glue', 'qnli']
    RTE = ['glue', 'rte']
    WNLI = ['glue', 'wnli']
    CB = ['super_glue', 'cb']
    BOOLQ = ['super_glue', 'boolq']
    COLA = ['glue', 'cola']


  class Model(Enum):
    FULLFT = 1
    LORA = 2
    IA3 = 3
    ADALORA = 4