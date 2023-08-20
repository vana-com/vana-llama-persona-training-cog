import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)

class LLMTrainer:
    def __init__(self, model_name, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, device_map):
        logging.info("Initializing the trainer with a model...")

        self.device_map = device_map

        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                logging.info("The GPU supports bfloat16: accelerate training with bf16=True.")

        # Load base model
        # TODO don't download from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        # TODO don't download from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)
        # pad_token_id = 32000
        # tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train_llm_model(
        self,
        train_path,
        validation_path,
        lora_r,
        lora_alpha,
        lora_dropout,
        output_dir,
        num_train_epochs,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        optim,
        save_steps,
        logging_steps,
        learning_rate,
        weight_decay,
        fp16,
        bf16,
        max_grad_norm,
        max_steps,
        warmup_ratio,
        group_by_length,
        lr_scheduler_type,
        new_model,
        max_seq_length,
        packing,
    ):
        logging.info("Loading the training data...")

        # Preprocessing and Data Loading
        datasets = load_dataset("json", data_files={"train": train_path, "validation": validation_path})

        logging.info("Configuring training...")

        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            # report_to="tensorboard"
        )

        # Training
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            dataset_text_field="text",
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=packing,
        )

        logging.info("Starting training...")

        trainer.train()

        logging.info("Training complete, saving the model...")

        trainer.model.save_pretrained(new_model)

        # I don't think we need to do this, if we just want to save the LoRA weights

        # # Reload model in FP16 and merge it with LoRA weights
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     low_cpu_mem_usage=True,
        #     return_dict=True,
        #     torch_dtype=torch.float16,
        #     device_map=self.device_map,
        # )
        # model = PeftModel.from_pretrained(base_model, new_model)
        # merged_model = model.merge_and_unload()

        # # Reload tokenizer to save it
        # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right"
        # model.resize_token_embeddings(len(tokenizer))

