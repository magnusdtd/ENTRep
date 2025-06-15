from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
  SentenceTransformerTrainingArguments,
  SentenceTransformerTrainer
)
from datasets import DatasetDict
from sentence_transformers import SentenceTransformer


class CLIP:
  def __init__(self, model: SentenceTransformer, finetuned_model_name: str):
    # define loss (note: loss expects columns to be ordered as anchor-positive-negative)
    self.loss = MultipleNegativesRankingLoss(model)
    self.model = model
    self.num_epochs = 2
    self.batch_size = 16
    self.lr = 1e-4

    self.train_args = SentenceTransformerTrainingArguments(
      output_dir=f"models/{finetuned_model_name}",
      num_train_epochs=self.num_epochs,
      per_device_train_batch_size=self.batch_size,
      per_device_eval_batch_size=self.batch_size,
      learning_rate=self.lr,
      # Evaluation settings
      eval_strategy="epoch",
      eval_steps=1,
      logging_steps=1,
    )

  def fine_tune(self, dataset:DatasetDict, evaluator_recall_train, evaluator_recall_valid):
    trainer = SentenceTransformerTrainer(
      model=self.model,
      args=self.train_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["valid"],
      loss=self.loss,
      evaluator=[evaluator_recall_train, evaluator_recall_valid]
    )
    trainer.train()