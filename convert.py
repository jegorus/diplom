import pandas as pd

class Converter:

  def get_experiment_name(exp_type, exp_num):
    return 'exp-' + exp_type + '-' + str(exp_num)

  def log_to_pandas(log_history):
    return pd.DataFrame.from_dict(log_history)
  
  @classmethod
  def pandas_to_csv(cls, pandas_df, exp_type, exp_num):
    exp_name = cls.get_experiment_name(exp_type, exp_num)
    pandas_df.to_csv(exp_name, index=False)
  
  @classmethod
  def log_to_csv(cls, log_history, exp_type, exp_num):
    df = cls.log_to_pandas(log_history)
    cls.pandas_to_csv(df, exp_type, exp_num)

  @classmethod
  def csv_to_pandas(cls, exp_type, exp_num):
    exp_name = cls.get_experiment_name(exp_type, exp_num)
    return pd.read_csv(exp_name)
  
  def prettify_results(pandas_df):
    pd.set_option('display.max_rows', None)
    ret_df = pandas_df[["epoch", "step", "loss", "eval_loss", "eval_accuracy", "eval_f1", "train_runtime"]].groupby(['step']).max()
    return ret_df.dropna().rename(columns={"loss" : "train_loss"})