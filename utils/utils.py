def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def print_trainable_named_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


import matplotlib.pyplot as plt

def make_train_eval_plot(name, steps_list, train_list, eval_list):

  # example: make_train_eval_plot('accuracy', df["step"], df["eval_f1"], df2["eval_f1"])

  plt.plot(steps_list, train_list, color='b', label='fullft')
  plt.plot(steps_list, eval_list, color='g', label='lora r=16, a=16')
  plt.ylabel(name, fontsize=16)
  plt.xlabel('steps', fontsize=14)
  plt.legend()
  plt.show()
