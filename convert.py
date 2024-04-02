import pandas as pd
from pandas.io.sql import PandasSQL

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
    return ret_df.dropna().rename(columns={"loss" : "train_loss"}).reset_index()
  
  def add_average_metric_column(pandas_df, name, metrics_list):
    
    def avg_metric(row):
      return sum([row[metric] for metric in metrics_list]) / len(metrics_list)

    pandas_df[name] = pandas_df.apply(avg_metric, axis=1)
    return pandas_df
  
  def create_df_max():
    return  pd.DataFrame(columns=["name", "max_acc", "max_f1", "max_avg", "epoch_max_acc", "epoch_max_f1", "epoch_max_avg"])

  @classmethod
  def add_row_to_df_max(cls, df_max, exp_type, exp_num):
    df = cls.csv_to_pandas(exp_type, exp_num)
    df = cls.prettify_results(df)
    df = cls.add_average_metric_column(df, 'eval_avg', ['eval_accuracy', 'eval_f1'])
    df_append = {"name" : cls.get_experiment_name(exp_type, exp_num)}

    df_tmp = df.iloc[df['eval_accuracy'].idxmax()]
    df_append["max_acc"] = df_tmp["eval_accuracy"]
    df_append["epoch_max_acc"] = df_tmp["epoch"]

    df_tmp = df.iloc[df['eval_f1'].idxmax()]
    df_append["max_f1"] = df_tmp["eval_f1"]
    df_append["epoch_max_f1"] = df_tmp["epoch"]

    df_tmp = df.iloc[df['eval_avg'].idxmax()]
    df_append["max_avg"] = df_tmp["eval_avg"]
    df_append["epoch_max_avg"] = df_tmp["epoch"]

    return df_max.append(df_append, ignore_index = True) 


  
