import gc
import logging
from typing import List
import torch
import json
import uuid
from cog import BasePredictor, Input, Path

from vanautils import FileManager

from scripts.train_llm_model import LLMTrainer
from training_config import *

from common import (
    clean_directories,
    extract_zip_and_flatten,
)

from scripts.process_chat_data import ChatDataProcessor

TRAINING_DIR = "/tmp/vana-llm-training/training-data"
CHECKPOINT_DIR = "/tmp/vana-llm-training/checkpoints"

logging.basicConfig(level=logging.INFO)

class Predictor(BasePredictor):
    def setup(self):
        logging.info("Setting up the Predictor...")
        self.file_manager = FileManager(download_dir=TRAINING_DIR)
        self.trainer = LLMTrainer(model_name, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, device_map)

    def predict(
        self,
        whatsapp_training_files: Path = Input(
            description="A ZIP file containing your training data (.txt WhatsApp chat logs, size not restricted). These chat logs contain your 'subject' that you want the trained model to embed in the output domain for later generating conversational text.",
            default=None
        ),
        whatsapp_training_files_urls: str = Input(
            description="A list of URLs that can be used instead of a zip file, e.g., [\"https://example.com/dataset/whatsapp.txt\"] Do not use both.",
            default=None
        ),
        subject_name: str = Input(description="The name of the subject to match in whatsapp_training_files_urls."),

        model_name: str = Input(description="The model to train from the Hugging Face hub.", default=model_name),

        lora_r: int = Input(description="LoRA attention dimension.", default=lora_r),
        lora_alpha: int = Input(description="Alpha parameter for LoRA scaling.", default=lora_alpha),
        lora_dropout: float = Input(description="Dropout probability for LoRA layers.", default=lora_dropout),
        num_train_epochs: int = Input(description="Number of training epochs.", default=num_train_epochs),
        fp16: bool = Input(description="Enable fp16 training.", default=fp16),
        bf16: bool = Input(description="Enable bf16 training (set bf16 to True with an A100).", default=bf16),
        per_device_train_batch_size: int = Input(description="Batch size per GPU for training.", default=per_device_train_batch_size),
        per_device_eval_batch_size: int = Input(description="Batch size per GPU for evaluation.", default=per_device_eval_batch_size),
        gradient_accumulation_steps: int = Input(description="Number of update steps to accumulate the gradients for.", default=gradient_accumulation_steps),
        gradient_checkpointing: bool = Input(description="Enable gradient checkpointing.", default=gradient_checkpointing),
        max_grad_norm: float = Input(description="Maximum gradient normal (gradient clipping).", default=max_grad_norm),
        learning_rate: float = Input(description="Initial learning rate (AdamW optimizer).", default=learning_rate),
        weight_decay: float = Input(description="Weight decay to apply to all layers except bias/LayerNorm weights.", default=weight_decay),
        optim: str = Input(description="Optimizer to use.", default=optim),
        lr_scheduler_type: str = Input(description="Learning rate schedule.", default=lr_scheduler_type),
        max_steps: int = Input(description="Number of training steps (overrides num_train_epochs).", default=max_steps),
        warmup_ratio: float = Input(description="Ratio of steps for a linear warmup (from 0 to learning rate).", default=warmup_ratio),
        group_by_length: bool = Input(description="Group sequences into batches with the same length.", default=group_by_length),
        save_steps: int = Input(description="Save checkpoint every X updates steps.", default=save_steps),
        logging_steps: int = Input(description="Log every X updates steps.", default=logging_steps),
        max_seq_length: int = Input(description="Maximum sequence length to use.", default=max_seq_length),
        packing: bool = Input(description="Pack multiple short examples in the same input sequence to increase efficiency.", default=packing),
    ) -> List[Path]:
        logging.info("Starting prediction process...")
        new_model_name = str(uuid.uuid4())

        # TODO: Safe to assume that this instance only processes one request at a time?

        if whatsapp_training_files is None and whatsapp_training_files_urls is None:
            logging.info("Downloading files from URLs...")
            raise Exception('No training data provided')

        # seed = 0

        clean_directories([TRAINING_DIR, CHECKPOINT_DIR])

        [
            self.file_manager.download_file(chat_url, preserve_structure=False) for chat_url in json.loads(whatsapp_training_files_urls)
        ] if whatsapp_training_files_urls is not None else None

        if whatsapp_training_files is not None:
            logging.info("Extracting zipped files...")
            extract_zip_and_flatten(whatsapp_training_files, TRAINING_DIR)

        logging.info("Processing chat data...")
        processor = ChatDataProcessor(
            train_split=0.9,
            filter_speaker=subject_name,
            no_overlap=True,
            final_format='colab'
        )
        train_file_name, validate_file_name, _ = processor.process_directory(TRAINING_DIR, TRAINING_DIR)

        params = {
            "model_name": model_name,
            "new_model_name": new_model_name,
            "train_path": f"{TRAINING_DIR}/{train_file_name}",
            "validation_path": f"{TRAINING_DIR}/{validate_file_name}",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "output_dir": CHECKPOINT_DIR,
            "num_train_epochs": num_train_epochs,
            "fp16": fp16,
            "bf16": bf16,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "max_grad_norm": max_grad_norm,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optim": optim,
            "lr_scheduler_type": lr_scheduler_type,
            "max_steps": max_steps,
            "warmup_ratio": warmup_ratio,
            "group_by_length": group_by_length,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "max_seq_length": max_seq_length,
            "packing": packing,
        }

        logging.info("Training the model...")
        self.trainer.train_llm_model(**params)

        logging.info("Collecting garbage and emptying CUDA cache...")
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: Evaluate whether it's better to use the final output rather than the newest checkpoint
        #
        # Assuming the checkpoint directory is structured as results/checkpoint-XXX/
        checkpoint_dirs = [d for d in Path(CHECKPOINT_DIR).glob("checkpoint-*")]
        latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(x.name.split('-')[-1]))

        # Collect the required files
        logging.info(f"Collecting generated files from {latest_checkpoint_dir}...")
        output_files = [latest_checkpoint_dir / 'adapter_model.bin',
                        latest_checkpoint_dir / 'adapter_config.json']

        # Add any other required files here

        logging.info("Prediction process completed.")
        return output_files

        # import zipfile
        # from pathlib import Path

        # # TODO figure out what to do with this
        # # Creating a ZIP file of all the files in the checkpoint directory
        # zip_path = Path(CHECKPOINT_DIR) / f'{latest_checkpoint_dir.name}.zip'
        # with zipfile.ZipFile(zip_path, 'w') as zipf:
        #     for file_path in latest_checkpoint_dir.iterdir():
        #         zipf.write(file_path, arcname=file_path.name)

        # return output_path

