from os import PathLike
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

import torch
import math


def load(path: PathLike):
    dataset = load_dataset("json", data_files=path)
    print(dataset)
    splits = dataset["train"].train_test_split(test_size=0.1)
    return splits


def initialize_model(model_path: PathLike):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./llama3.3_finetuned",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        save_total_limit=2,
    )

    return config, model, tokenizer, training_args


def preprocess_function_builder(tokenizer):
    def preprocess_function(examples):
        inputs = [ex["prompt"] for ex in examples]
        targets = [ex["response"] for ex in examples]
        model_inputs = tokenizer(
            inputs, padding="max_length", truncation=True, max_length=512
        )
        labels = tokenizer(
            targets, padding="max_length", truncation=True, max_length=512
        ).input_ids
        model_inputs["labels"] = labels
        return model_inputs

    return preprocess_function


def main():
    data = load("/Users/sanderhergarten/datasources/bookhelper/dataset/results.json")
    model_config, model, tokenizer, training_args = initialize_model(
        "meta-llama/Llama-3.3-70B-Instruct"
    )
    preprocess_function = preprocess_function_builder(tokenizer)

    tokenized_datasets = data.map(preprocess_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # Save the model and tokenizer
    model.save_pretrained("path_to_save_model")
    tokenizer.save_pretrained("path_to_save_tokenizer")


if __name__ == "__main__":
    main()
