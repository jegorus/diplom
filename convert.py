import pandas as pd
from pandas.io.sql import PandasSQL
from torch._C import _export_opnames

class Converter:

  def get_experiment_name(exp_type, exp_num):
    return 'exp-' + exp_type + '-' + str(exp_num)

  def log_to_pandas(log_history):
    return pd.DataFrame.from_dict(log_history)
  
  @classmethod
  def pandas_to_csv(cls, pandas_df, exp_type, exp_num):
    exp_name = cls.get_experiment_name(exp_type, exp_num)
    cls.pandas_to_csv_name(pandas_df, exp_name)
  
  @classmethod
  def log_to_csv(cls, log_history, exp_type, exp_num):
    exp_name = cls.get_experiment_name(exp_type, exp_num)
    cls.log_to_csv_name(log_history, exp_name)

  @classmethod
  def csv_to_pandas(cls, exp_type, exp_num):
    exp_name = cls.get_experiment_name(exp_type, exp_num)
    return cls.csv_to_pandas_name(exp_name)

  
  def csv_to_pandas_name(name):
    return pd.read_csv(name)
  
  def pandas_to_csv_name(pandas_df, name):
    pandas_df.to_csv(name, index=False)
  
  @classmethod
  def log_to_csv_name(cls, log_history, name):
    df = cls.log_to_pandas(log_history)
    cls.pandas_to_csv_name(df, name)


  def prettify_results(pandas_df, metrics_list = ["acc", "f1"]):
    pd.set_option('display.max_rows', None)
    pandas_df = pandas_df.rename(columns={"eval_accuracy": "eval_acc"})
    metrics_list = ["eval_" + x for x in metrics_list]
    ret_df = pandas_df[["epoch", "step", "loss", "eval_loss"] + metrics_list +  ["train_runtime"]].groupby(['step']).max()
    return ret_df.dropna().rename(columns={"loss" : "train_loss"}).reset_index()
  

  def add_average_metric_column(pandas_df, name, metrics_list):
    
    metrics_list = ["eval_" + x for x in metrics_list]

    def avg_metric(row):
      return sum([row[metric] for metric in metrics_list]) / len(metrics_list)

    pandas_df[name] = pandas_df.apply(avg_metric, axis=1)
    return pandas_df
  

  def create_df_max(metrics_list = ["acc", "f1"], extra_columns=[]):
    columns = []
    columns += ["max_" + x for x in metrics_list]
    if len(metrics_list) > 1:
      columns += ["max_avg"]
    columns += ["epoch_" + x for x in columns]
    return  pd.DataFrame(columns=["name"] + columns + extra_columns)

  @classmethod
  def add_row_to_df_max(cls, df_max, exp_type, exp_num, metrics_list = ["acc", "f1"]):
    df1 = cls.csv_to_pandas(exp_type, exp_num)
    df = cls.prettify_results(df1, metrics_list)
    df = cls.add_average_metric_column(df, 'eval_avg', metrics_list)
    df_append = {"name" : cls.get_experiment_name(exp_type, exp_num)}

    for metric in metrics_list:
      df_tmp = df.iloc[df["eval_" + metric].idxmax()]
      df_append["max_" + metric] = df_tmp["eval_" + metric]
      df_append["epoch_max_" + metric] = df_tmp["epoch"]

    df_tmp = df.iloc[df['eval_avg'].idxmax()]
    df_append["max_avg"] = df_tmp["eval_avg"]
    df_append["epoch_max_avg"] = df_tmp["epoch"]

    df_append["train_runtime"] = df1.iloc[len(df1) - 1]["train_runtime"]

    df_max.loc[len(df_max)] = df_append 
    return df_max
  
  def create_df_exp_results():
     return pd.DataFrame(columns=["name", "acc_avg", "f1_avg", "both_avg", "acc_max", "f1_max", "both_max", "seeds_list", "epoch_acc_avg", "epoch_f1_avg", "epoch_both_avg"])
  
  def add_row_to_df_exp_results(df_max, df_exp_results, name, i_begin, i_end, seeds_list):
    df_current = df_max.iloc[i_begin:i_end]
    num = len(df_exp_results)
    df_exp_results.loc[len(df_exp_results)] = pd.Series()
    df_exp_results["name"].iloc[num] = name
    df_exp_results["acc_avg"].iloc[num] = df_current["max_acc"].mean()
    df_exp_results["f1_avg"].iloc[num] = df_current["max_f1"].mean()
    df_exp_results["both_avg"].iloc[num] = df_current["max_avg"].mean()
    df_exp_results["acc_max"].iloc[num] = df_current["max_acc"].max()
    df_exp_results["f1_max"].iloc[num] = df_current["max_f1"].max()
    df_exp_results["both_max"].iloc[num] = df_current["max_avg"].max()
    df_exp_results["seeds_list"].iloc[num] = seeds_list
    df_exp_results["epoch_acc_avg"].iloc[num] = df_current["epoch_max_acc"].mean()
    df_exp_results["epoch_f1_avg"].iloc[num] = df_current["epoch_max_f1"].mean()
    df_exp_results["epoch_both_avg"].iloc[num] = df_current["epoch_max_avg"].mean()

    return df_exp_results

  def create_df_info_to_results_mapper():
    # used in df_exp_results
    return pd.DataFrame(columns=["name", "i_begin", "i_end", "seeds_list"])
  
  def add_row_to_df_info_to_results_mapper(df_info_to_results_mapper, name, i_begin, i_end, seeds_list):
    df_info_to_results_mapper.loc[len(df_info_to_results_mapper)] = {
    "name": name,
    "i_begin": i_begin,
    "i_end": i_end,
    "seeds_list": seeds_list,
    }
    return df_info_to_results_mapper


  
