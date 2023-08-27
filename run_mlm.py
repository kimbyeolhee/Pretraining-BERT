import argparse
import logging
import os

import datasets
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.get_logger import get_logger

logger = get_logger()

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
# https://colab.research.google.com/drive/1An1VNpKKMRVrwcdQQNSe7Omh_fl2Gj-2?usp=sharing#scrollTo=5Pe5ZkpvVBl1


def main(config):
    # Load dataset
    logger.info("##### Loading dataset #####")
    raw_datasets = load_dataset(
        "text", 
        data_files=config.data.path
        )
    raw_datasets["train"] = load_dataset(
        "text",
        data_files=config.data.path,
        split=f"train[{config.data.validation_split_percentage}%:]"
    )
    raw_datasets["validation"] = load_dataset(
        "text",
        data_files=config.data.path,
        split=f"train[:{config.data.validation_split_percentage}%]"
    )
    logger.info("length of train dataset: {}".format(len(raw_datasets["train"])))
    logger.info("length of validation dataset: {}".format(len(raw_datasets["validation"])))

    # Load pretrained model and tokenizer
    logger.info("##### Loading pretrained model and tokenizer #####")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    model = AutoModelForMaskedLM.from_pretrained(config.model.name)

    # Tokenize dataset
    logger.info("##### Tokenizing dataset #####")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length",
                         max_length=config.tokenizer.max_length, return_special_tokens_mask=True, return_tensors="pt")
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
    )

    # Data collator
    logger.info("##### Data collator #####")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=config.data.mlm_probability
    )

    if not os.path.exists(config.training.output_dir):
        os.makedirs(config.training.output_dir)

    # Initialize Trainer
    logger.info("##### Initialize Trainer #####")
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        save_total_limit=config.training.save_total_limit,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        evaluation_strategy=config.training.evaluation_strategy, # "epoch", "steps"
        save_strategy=config.training.save_strategy, # "epoch", "steps"
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        logging_dir=config.training.logging_dir,
        logging_steps=1000,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Training
    logger.info("##### Training #####")
    metrics = trainer.train()
    logger.info("Training metrics: {}".format(metrics))

    # Evaluation
    logger.info("##### Evaluation #####")
    metrics = trainer.evaluate()
    logger.info("Evaluation metrics: {}".format(metrics))

    # Save model
    logger.info("##### Save model #####")
    trainer.save_model(config.training.output_dir)




if __name__ == "__main__":
    root_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(root_path, "configs", args.config + ".yaml"))

    logger.info(f"Config: {config}")

    main(config)
