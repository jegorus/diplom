# https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/trainer.py#L3298 - оригинальный evaluate с huggingface
# https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train

import time
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
import math



class CustomTrainer(Trainer):

      def evaluate(
          self,
          eval_dataset: Optional[Dataset] = None,
          ignore_keys: Optional[List[str]] = None,
          metric_key_prefix: str = "eval",
      ) -> Dict[str, float]:

          # memory metrics - must set up as early as possible
          self._memory_tracker.start()

          eval_dataloader = self.get_eval_dataloader(eval_dataset)
          train_dataloader = self.get_train_dataloader()

          eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
          eval_output = eval_loop(
              eval_dataloader,
              description="Evaluation",
              # No point gathering the predictions if there are no metrics, otherwise we defer to
              # self.args.prediction_loss_only
              prediction_loss_only=True if self.compute_metrics is None else None,
              ignore_keys=ignore_keys,
              metric_key_prefix=metric_key_prefix,
          )


          eval_output.metrics.update(
              {"train_runtime": round(time.time() - my_custom_start_time, 4)}
          )

          self.log(eval_output.metrics)

          self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_output.metrics)

          self._memory_tracker.stop_and_update_metrics(eval_output.metrics)


          # only works in Python >= 3.9
          return  eval_output.metrics