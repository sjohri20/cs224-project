import argparse
import yaml
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import csv
import os
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import AutoTokenizer
import wandb
from torch.utils.data import Dataset, DataLoader
from src.models import ModelWithRegressorForLM
from src.utils import run_validation_with_save, save_generation_to_file

def save_step_data_to_csv(regression_outputs, initial_length, final_length, global_step, epoch, sample_idx, csv_dir="logs/token_data"):
    """
    Save detailed per-step and per-layer metrics to CSV files.
    Args:
        regression_outputs: List of dictionaries with regression outputs for each step
        initial_length: Starting token count (prompt length)
        final_length: Final token count after generation
        global_step: Current global training step
        epoch: Current epoch
        sample_idx: Index of the current training sample
        csv_dir: Directory to save CSV files
    """
    # Create directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # Filename format: epoch_step_sample.csv
    filename = f"{csv_dir}/epoch_{epoch}_step_{global_step}_sample_{sample_idx}.csv"
    
    num_layers = len([k for k in regression_outputs[0].keys() if k.startswith("layer_")])
    
    # Prepare data for CSV
    rows = []
    
    for step_idx, step_preds in enumerate(regression_outputs):
        # Current position in the sequence
        current_position = initial_length + step_idx
        # Remaining tokens to be generated (target value)
        remaining_tokens = final_length - current_position
        
        # Basic row information
        row = {
            'epoch': epoch,
            'global_step': global_step,
            'sample_idx': sample_idx,
            'token_position': current_position,
            'remaining_tokens': remaining_tokens
        }
        
        # Add predictions from each layer
        for layer, pred in step_preds.items():
            pred_value = pred.item() if isinstance(pred.item(), (int, float)) else pred.item()[0]
            row[f"{layer}_prediction"] = pred_value
            row[f"{layer}_error"] = abs(pred_value - remaining_tokens)
        
        rows.append(row)
    
    # Write to CSV
    if rows:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    
    return filename

def log_simplified_metrics_to_wandb(regression_outputs, initial_length, final_length, global_step, epoch, frozen_layers=None):
    """
    Log only start and mid training metrics to wandb with simplified naming.
    Only logs for active (non-frozen) layers.
    Args:
        regression_outputs: List of dictionaries with regression outputs for each step
        initial_length: Starting token count (prompt length)
        final_length: Final token count after generation
        global_step: Current global training step 
        epoch: Current epoch
        frozen_layers: Dictionary indicating which layers are frozen
    """
    if not regression_outputs or len(regression_outputs) == 0:
        return
    
    if frozen_layers is None:
        frozen_layers = {}
    
    num_steps = len(regression_outputs)
    
    # Only log metrics at the beginning and midpoint
    positions = []
    position_names = []
    
    # Beginning (first token)
    if num_steps > 0:
        positions.append(0)
        position_names.append("start")
    
    # Midpoint token position
    if num_steps > 1:
        midpoint_idx = num_steps // 2
        positions.append(midpoint_idx)
        position_names.append("mid")
    
    # Prepare metrics to log
    metrics_to_log = {}
    
    # Extract metrics for each position
    for idx, (pos_idx, pos_name) in enumerate(zip(positions, position_names)):
        if pos_idx < len(regression_outputs):
            step_preds = regression_outputs[pos_idx]
            current_position = initial_length + pos_idx
            remaining_tokens = final_length - current_position
            
            # Calculate error for each layer
            for layer, pred in step_preds.items():
                # Skip frozen layers
                layer_idx = int(layer.split('_')[1])
                if frozen_layers.get(f"layer_{layer_idx}", False):
                    continue
                
                pred_value = pred.item() if isinstance(pred.item(), (int, float)) else pred.item()[0]
                error = abs(pred_value - remaining_tokens)
                
                # Use simplified naming convention
                metrics_to_log[f"train_layer_{layer_idx}_loss_{pos_name}"] = error
    
    # Add total generated length
    metrics_to_log["total_generated_length"] = final_length
    
    # Log to wandb if we have metrics
    if metrics_to_log:
        metrics_to_log["step"] = global_step
        metrics_to_log["epoch"] = epoch
        wandb.log(metrics_to_log)

def save_batch_step_data_to_csv(regression_outputs, initial_lengths, final_lengths, global_step, epoch, sample_indices, csv_dir="logs/token_data"):
    """
    Save detailed per-step and per-layer metrics for a batch of prompts to CSV files.
    Args:
        regression_outputs: List of dictionaries with regression outputs for each step
        initial_lengths: List/tensor of starting token counts (prompt lengths) for each item in batch
        final_lengths: List/tensor of final token counts after generation for each item in batch  
        global_step: Current global training step
        epoch: Current epoch
        sample_indices: List of indices of the current training samples
        csv_dir: Directory to save CSV files
    
    Returns:
        List of saved CSV filenames
    """
    # Create directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    batch_size = len(sample_indices)
    csv_filenames = []
    
    # Process each item in the batch
    for batch_idx in range(batch_size):
        # Get data for this batch item
        initial_length = initial_lengths[batch_idx]
        final_length = final_lengths[batch_idx]
        sample_idx = sample_indices[batch_idx]
        
        # Filename format: epoch_step_sample.csv
        filename = f"{csv_dir}/epoch_{epoch}_step_{global_step}_sample_{sample_idx}.csv"
        csv_filenames.append(filename)
        
        num_layers = len([k for k in regression_outputs[0].keys() if k.startswith("layer_")])
        
        # Prepare data for CSV
        rows = []
        
        for step_idx, step_preds in enumerate(regression_outputs):
            # Current position in the sequence
            current_position = initial_length + step_idx
            
            # Skip if we've gone beyond the final length for this sequence
            if current_position >= final_length:
                continue
                
            # Remaining tokens to be generated (target value)
            remaining_tokens = final_length - current_position
            
            # Basic row information
            row = {
                'epoch': epoch,
                'global_step': global_step,
                'sample_idx': sample_idx,
                'token_position': current_position,
                'remaining_tokens': remaining_tokens
            }
            
            # Add predictions from each layer
            for layer, preds in step_preds.items():
                # Get the prediction for this batch item
                pred = preds[batch_idx]
                pred_value = pred.item() if isinstance(pred, torch.Tensor) else pred
                row[f"{layer}_prediction"] = pred_value
                row[f"{layer}_error"] = abs(pred_value - remaining_tokens)
            
            rows.append(row)
        
        # Write to CSV
        if rows:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
    
    return csv_filenames

def log_batch_simplified_metrics_to_wandb(regression_outputs, initial_lengths, final_lengths, global_step, epoch, frozen_layers=None):
    """
    Log only start and mid training metrics to wandb with simplified naming for a batch of sequences.
    Only logs for active (non-frozen) layers.
    Args:
        regression_outputs: List of dictionaries with regression outputs for each step
        initial_lengths: List/tensor of starting token counts for each item in batch
        final_lengths: List/tensor of final token counts for each item in batch
        global_step: Current global training step 
        epoch: Current epoch
        frozen_layers: Dictionary indicating which layers are frozen
    """
    if not regression_outputs or len(regression_outputs) == 0:
        return
    
    if frozen_layers is None:
        frozen_layers = {}
    
    batch_size = len(initial_lengths)
    
    # Collect metrics across the batch
    batch_metrics = {}
    
    # Process each sequence in the batch
    for batch_idx in range(batch_size):
        initial_length = initial_lengths[batch_idx]
        final_length = final_lengths[batch_idx]
        
        num_steps = sum(1 for step_idx, _ in enumerate(regression_outputs) 
                       if initial_length + step_idx < final_length)
        
        # Skip if no valid steps
        if num_steps == 0:
            continue
            
        # Only log metrics at the beginning and midpoint
        positions = []
        position_names = []
        
        # Beginning (first token)
        if num_steps > 0:
            positions.append(0)
            position_names.append("start")
        
        # Midpoint token position
        if num_steps > 1:
            midpoint_idx = num_steps // 2
            positions.append(midpoint_idx)
            position_names.append("mid")
        
        # Extract metrics for each position
        for pos_idx, pos_name in zip(positions, position_names):
            if pos_idx < len(regression_outputs):
                step_preds = regression_outputs[pos_idx]
                current_position = initial_length + pos_idx
                
                # Skip if this position exceeds the final length for this batch item
                if current_position >= final_length:
                    continue
                    
                remaining_tokens = final_length - current_position
                
                # Calculate error for each layer
                for layer, preds in step_preds.items():
                    # Skip frozen layers
                    layer_idx = int(layer.split('_')[1])
                    if frozen_layers.get(f"layer_{layer_idx}", False):
                        continue
                    
                    # Get prediction for this batch item
                    pred = preds[batch_idx]
                    pred_value = pred.item() if isinstance(pred, torch.Tensor) else pred
                    error = abs(pred_value - remaining_tokens)
                    
                    # Track error in the batch metrics
                    metric_key = f"train_layer_{layer_idx}_loss_{pos_name}"
                    if metric_key not in batch_metrics:
                        batch_metrics[metric_key] = []
                    batch_metrics[metric_key].append(error)
    
    # Compute average metrics across the batch
    metrics_to_log = {
        "total_generated_length": sum(final_lengths) / batch_size,
        "step": global_step,
        "epoch": epoch
    }
    
    # Add average metrics across the batch
    for metric_key, values in batch_metrics.items():
        if values:  # Only log non-empty metrics
            metrics_to_log[metric_key] = sum(values) / len(values)
    
    # Log to wandb if we have metrics
    if len(metrics_to_log) > 3:  # More than just step, epoch, and length
        wandb.log(metrics_to_log)

class PromptDataset(Dataset):
    """
    Dataset class for prompt data loaded from CSV files.
    """
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file with prompts
        """
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.data.iloc[idx]['prompt'],
            'idx': idx
        }

def main():
    parser = argparse.ArgumentParser(description="Train regression heads to predict output token count.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()
    
    # Load configuration from YAML file.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
    train_prompt_file = config.get("prompt_train")
    val_prompt_file = config.get("prompt_val")
    max_new_tokens = config.get("max_new_tokens", 10)
    beam_width = config.get("beam_width", 1)
    top_k = config.get("top_k", 0)
    top_p = config.get("top_p", 0.0)
    include_token_count = config.get("include_token_count", False)
    learning_rate = config.get("lr", 1e-4)
    num_epochs = config.get("epochs", 1)
    early_stopping = config.get("early_stopping", 5)
    val_freq = config.get("val_freq", 0)  # 0 means validate only at end of epoch
    save_path = config.get("save_path", "regression_model.pt")
    batch_size = config.get("batch_size", 1)  # Default batch size is 1
    num_workers = config.get("num_workers", 0)  # Number of worker threads for DataLoader
    
    # Output files for generated text
    train_outputs_file = config.get("train_outputs_file", "outputs/train_generations.jsonl")
    val_outputs_file = config.get("val_outputs_file", "outputs/val_generations.jsonl")
    
    # CSV log directory
    csv_log_dir = config.get("csv_log_dir", "logs/token_data")
    
    # Scheduler parameters
    scheduler_type = config.get("scheduler_type", "cosine")
    T_max = config.get("T_max", num_epochs)  # Default to number of epochs
    eta_min = config.get("eta_min", 0)
    warmup_steps = config.get("warmup_steps", 0)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ModelWithRegressorForLM(model_name, include_token_count=include_token_count).to(device)

    wandb.init(project=config.get("wandb_project"), 
               entity = config.get("wandb_entity"), 
               config=config)
    print("Wandb initialized.")
    wandb.watch(model, log="gradients")

    # Create datasets
    train_dataset = PromptDataset(train_prompt_file)
    val_dataset = PromptDataset(val_prompt_file)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Enable random shuffling of the training data
        num_workers=num_workers
    )
    
    # For validation, we don't need shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Loaded {len(train_dataset)} training prompts from {train_prompt_file}")
    print(f"Loaded {len(val_dataset)} validation prompts from {val_prompt_file}")
    
    # Freeze LM parameters.
    for param in model.model.parameters():
        param.requires_grad = False
    model.model.eval()
    print("Language model frozen, only regression heads will be trained")
    
    optimizer = torch.optim.Adam(model.regression_heads.parameters(), lr=learning_rate)
    
    # Set up the scheduler
    total_steps = len(train_dataset) * num_epochs
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"Using cosine scheduler with T_max={T_max}, eta_min={eta_min}")
    elif scheduler_type == "cosine_warm_restarts":
        T_0 = config.get("T_0", T_max)  # First restart cycle length
        T_mult = config.get("T_mult", 1)  # Cycle length multiplier
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        print(f"Using cosine warm restarts with T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
    else:
        scheduler = None
        print("No LR scheduler used")

    # Track best losses per layer
    best_overall_val_loss = float('inf')
    best_layer_val_losses = {f"layer_{i}": float('inf') for i in range(model.num_layers)}
    # Track early stopping counters per layer
    layer_early_stopping_counters = {f"layer_{i}": 0 for i in range(model.num_layers)}
    # Track which layers are frozen
    frozen_layers = {f"layer_{i}": False for i in range(model.num_layers)}
    
    # Create directory for layer-specific models
    os.makedirs("layer_models", exist_ok=True)
    
    # Initialize tracking metrics table
    headers = ["Epoch", "Step", "LR"] + \
              [f"Train_L{i}" for i in range(model.num_layers)] + \
              [f"Val_L{i}" for i in range(model.num_layers)] + \
              [f"Frozen_L{i}" for i in range(model.num_layers)]
    print(" | ".join(headers))
    
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()  # Set model (regression heads) to training mode
        total_train_loss = 0.0
        train_layer_losses = {f"layer_{i}": 0.0 for i in range(model.num_layers)}
        samples_since_validation = 0
        
        # Check if all layers are frozen
        if all(frozen_layers.values()):
            print("All regression heads are frozen. Stopping training.")
            break
            
        # Use the DataLoader to iterate through the training data (with automatic shuffling)
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            prompts = batch['prompt']  # Get all prompts from the batch
            sample_indices = batch['idx'].tolist()  # Get the original indices for logging
            
            # Print the first prompt as an example
            print(f"Processing batch {batch_idx+1}/{len(train_loader)} with {len(prompts)} prompts. First prompt: {prompts[0][:50]}...")
            
            # Before training, freeze gradient flow for frozen layers
            for layer_idx in range(model.num_layers):
                layer_name = f"layer_{layer_idx}"
                if frozen_layers[layer_name]:
                    for param in model.regression_heads[layer_idx].parameters():
                        param.requires_grad = False
                else:
                    for param in model.regression_heads[layer_idx].parameters():
                        param.requires_grad = True
            
            # Tokenize all prompts in the batch
            inputs = tokenizer(prompts.tolist(), padding=True, return_tensors="pt").to(device)
            initial_lengths = inputs["input_ids"].sum(dim=1).tolist()  # Get the actual lengths (ignoring padding)
            
            # Train on the batch of prompts
            generated_ids_batch, regression_outputs_batch, loss, layer_losses, step_layer_losses, gradient_data = model.train_regressor_on_prompt_batch(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                optimizer,
                beam_width=beam_width,
                top_k=top_k,
                top_p=top_p
            )
            
            # Get the final lengths for each sequence in the batch
            final_lengths = [seq.size(1) for seq in generated_ids_batch.split(1)]
            
            # Decode generated texts
            generated_texts = [tokenizer.decode(ids[0]) for ids in generated_ids_batch.split(1)]
            
            # Log total loss for this batch
            total_train_loss += loss
            
            # Log simplified metrics to wandb (beginning and midpoint only)
            log_batch_simplified_metrics_to_wandb(
                regression_outputs_batch, 
                initial_lengths, 
                final_lengths,
                global_step,
                epoch + 1,
                frozen_layers
            )
            
            # Save detailed per-token data to CSV for each prompt in the batch
            csv_filenames = save_batch_step_data_to_csv(
                regression_outputs_batch,
                initial_lengths,
                final_lengths,
                global_step,
                epoch + 1,
                sample_indices,
                csv_dir=csv_log_dir
            )
            print(f"Saved detailed token data to {len(csv_filenames)} CSV files")
            
            # Save generated text for each prompt to file
            for i, (prompt, generated_text, final_length, sample_idx, csv_filename) in enumerate(zip(
                prompts, generated_texts, final_lengths, sample_indices, csv_filenames
            )):
                generation_data = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "prompt_idx_in_batch": i,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "token_count": final_length,
                    "loss": loss,  # Overall batch loss
                    "layer_losses": {k: v for k, v in layer_losses.items()},
                    "frozen_layers": {k: v for k, v in frozen_layers.items()},
                    "timestamp": datetime.now().isoformat(),
                    "csv_data_file": csv_filename
                }
                save_generation_to_file(train_outputs_file, generation_data)
            
            # Accumulate layer losses only for non-frozen layers
            for layer, val in layer_losses.items():
                if not frozen_layers[layer]:
                    train_layer_losses[layer] += val
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            avg_length = sum(final_lengths) / len(final_lengths)
            print(f"  Batch average tokens: {avg_length:.1f}. Loss: {loss:.4f}, LR: {current_lr:.6f}")
            
            # Log only learning rate to wandb
            wandb.log({
                "lr": current_lr,
                "step": global_step,
                "epoch": epoch + 1,
                "batch_size": len(prompts)
            })
            
            samples_since_validation += len(prompts)
            
            # Run validation at specified frequency if val_freq > 0
            if val_freq > 0 and samples_since_validation >= val_freq:
                print(f"\nRunning validation after {samples_since_validation} training samples...")
                samples_since_validation = 0
                
                # Current training stats
                active_training_layers = [layer for layer, frozen in frozen_layers.items() if not frozen]
                
                if not active_training_layers:
                    print("All regression heads are frozen. Stopping training.")
                    break
                    
                # Calculate average training loss for active layers
                curr_avg_train_loss = 0.0
                count_active = 0
                curr_avg_train_layer_losses = {}
                
                for layer, loss_sum in train_layer_losses.items():
                    if not frozen_layers[layer]:
                        # Only count samples where the layer was active
                        samples_active = batch_idx + 1
                        avg_loss = loss_sum / samples_active
                        curr_avg_train_layer_losses[layer] = avg_loss
                        curr_avg_train_loss += avg_loss
                        count_active += 1
                
                if count_active > 0:
                    curr_avg_train_loss /= count_active
                
                # Run validation with the validation DataLoader
                avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                    model, tokenizer, val_loader, device, 
                    max_new_tokens, beam_width, top_k, top_p,
                    val_outputs_file, epoch, global_step,
                    csv_log_dir=csv_log_dir,
                    frozen_layers=frozen_layers
                )
                
                # Log metrics table row
                metrics_row = [f"{epoch+1}", f"{global_step}", f"{current_lr:.6f}"]
                for i in range(model.num_layers):
                    layer_key = f"layer_{i}"
                    if frozen_layers[layer_key]:
                        metrics_row.append("frozen")
                    else:
                        metrics_row.append(f"{curr_avg_train_layer_losses.get(layer_key, 0):.4f}")
                for i in range(model.num_layers):
                    layer_key = f"layer_{i}"
                    metrics_row.append(f"{avg_val_layer_losses.get(layer_key, 0):.4f}")
                for i in range(model.num_layers):
                    layer_key = f"layer_{i}"
                    metrics_row.append(f"{'Yes' if frozen_layers[layer_key] else 'No'}")
                print(" | ".join(metrics_row))
                
                # Check for overall model improvement
                if avg_val_loss < best_overall_val_loss:
                    best_overall_val_loss = avg_val_loss
                    print(f"New best overall validation loss! Saving complete model to {save_path}")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'config': config,
                        'epoch': epoch,
                        'step': global_step,
                        'loss': best_overall_val_loss,
                        'frozen_layers': frozen_layers,
                    }, save_path)
                
                # Check for per-layer model improvement or early stopping
                improved_layers = []
                layers_not_improved = []
                
                for layer, val_loss in avg_val_layer_losses.items():
                    # Skip already frozen layers
                    if frozen_layers[layer]:
                        continue
                        
                    if val_loss < best_layer_val_losses[layer]:
                        best_layer_val_losses[layer] = val_loss
                        improved_layers.append(layer)
                        # Reset early stopping counter for this layer
                        layer_early_stopping_counters[layer] = 0
                        
                        # Save individual layer model
                        layer_idx = int(layer.split('_')[1])
                        layer_model_path = f"layer_models/layer_{layer_idx}_model.pt"
                        
                        # Save only the relevant regression head
                        torch.save({
                            'regression_head': model.regression_heads[layer_idx].state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'step': global_step,
                            'loss': val_loss,
                        }, layer_model_path)
                        print(f"New best validation loss for {layer}! Saved to {layer_model_path}")
                    else:
                        # Increment early stopping counter for this layer
                        layer_early_stopping_counters[layer] += 1
                        layers_not_improved.append(f"{layer} ({layer_early_stopping_counters[layer]}/{early_stopping})")
                        
                        # Check if this layer should be frozen
                        if layer_early_stopping_counters[layer] >= early_stopping:
                            frozen_layers[layer] = True
                            print(f"Early stopping triggered for {layer}. Freezing this regression head.")
                
                if improved_layers:
                    print(f"Improved layers in this step: {', '.join(improved_layers)}")
                if layers_not_improved:
                    print(f"Layers without improvement: {', '.join(layers_not_improved)}")
                
                # Check if all layers are frozen
                if all(frozen_layers.values()):
                    print(f"All regression heads are frozen. Training stopped after {batch_idx+1} samples in epoch {epoch+1}")
                    break
                
                # Resume training
                model.train()
            
            # Step the scheduler based on iteration
            if scheduler and scheduler_type == "cosine_warm_restarts":
                scheduler.step(epoch + batch_idx / len(train_loader))
        
        # Check if all layers are frozen
        if all(frozen_layers.values()):
            print(f"All regression heads are frozen. Training stopped after epoch {epoch+1}")
            break
            
        # Run validation at the end of each epoch regardless of val_freq
        if val_freq == 0 or samples_since_validation > 0:
            # Calculate average training losses for the full epoch
            active_training_layers = [layer for layer, frozen in frozen_layers.items() if not frozen]
            
            if not active_training_layers:
                print("All regression heads are frozen. Stopping training.")
                break
                
            # Calculate average training loss for active layers
            avg_train_loss = 0.0
            count_active = 0
            avg_train_layer_losses = {}
            
            for layer, loss_sum in train_layer_losses.items():
                if not frozen_layers[layer]:
                    # Get count of samples where this layer was active
                    samples_active = len(train_loader)
                    avg_loss = loss_sum / samples_active
                    avg_train_layer_losses[layer] = avg_loss
                    avg_train_loss += avg_loss
                    count_active += 1
            
            if count_active > 0:
                avg_train_loss /= count_active
            
            print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
            
            # Validation phase with the validation DataLoader
            avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                model, tokenizer, val_loader, device, 
                max_new_tokens, beam_width, top_k, top_p,
                val_outputs_file, epoch, global_step,
                csv_log_dir=csv_log_dir,
                frozen_layers=frozen_layers
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics table row
            metrics_row = [f"{epoch+1}", f"{global_step}", f"{current_lr:.6f}"]
            for i in range(model.num_layers):
                layer_key = f"layer_{i}"
                if frozen_layers[layer_key]:
                    metrics_row.append("frozen")
                else:
                    metrics_row.append(f"{avg_train_layer_losses.get(layer_key, 0):.4f}")
            for i in range(model.num_layers):
                layer_key = f"layer_{i}"
                metrics_row.append(f"{avg_val_layer_losses.get(layer_key, 0):.4f}")
            for i in range(model.num_layers):
                layer_key = f"layer_{i}"
                metrics_row.append(f"{'Yes' if frozen_layers[layer_key] else 'No'}")
            print(" | ".join(metrics_row))
            
            # Check for overall model improvement
            if avg_val_loss < best_overall_val_loss:
                best_overall_val_loss = avg_val_loss
                print(f"New best overall validation loss! Saving complete model to {save_path}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config,
                    'epoch': epoch,
                    'step': global_step,
                    'loss': best_overall_val_loss,
                    'frozen_layers': frozen_layers,
                }, save_path)
            
            # Check for per-layer model improvement
            improved_layers = []
            layers_not_improved = []
            
            for layer, val_loss in avg_val_layer_losses.items():
                # Skip already frozen layers
                if frozen_layers[layer]:
                    continue
                    
                if val_loss < best_layer_val_losses[layer]:
                    best_layer_val_losses[layer] = val_loss
                    improved_layers.append(layer)
                    # Reset early stopping counter for this layer
                    layer_early_stopping_counters[layer] = 0
                    
                    # Save individual layer model
                    layer_idx = int(layer.split('_')[1])
                    layer_model_path = f"layer_models/layer_{layer_idx}_model.pt"
                    
                    # Save only the relevant regression head
                    torch.save({
                        'regression_head': model.regression_heads[layer_idx].state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'step': global_step,
                        'loss': val_loss,
                    }, layer_model_path)
                    print(f"New best validation loss for {layer}! Saved to {layer_model_path}")
                else:
                    # Increment early stopping counter for this layer
                    layer_early_stopping_counters[layer] += 1
                    layers_not_improved.append(f"{layer} ({layer_early_stopping_counters[layer]}/{early_stopping})")
                    
                    # Check if this layer should be frozen
                    if layer_early_stopping_counters[layer] >= early_stopping:
                        frozen_layers[layer] = True
                        print(f"Early stopping triggered for {layer}. Freezing this regression head.")
            
            if improved_layers:
                print(f"Improved layers in this epoch: {', '.join(improved_layers)}")
            if layers_not_improved:
                print(f"Layers without improvement: {', '.join(layers_not_improved)}")
            
            # Check if all layers are frozen
            if all(frozen_layers.values()):
                print(f"All regression heads are frozen. Training stopped after epoch {epoch+1}")
                break
        
        # Step the scheduler based on epoch
        if scheduler and scheduler_type == "cosine":
            scheduler.step()
    
    print(f"Training complete. Best overall validation loss: {best_overall_val_loss:.4f}")
    print("Best validation losses per layer:")
    for layer, loss in best_layer_val_losses.items():
        print(f"  {layer}: {loss:.4f}")
    print("Final frozen status per layer:")
    for layer, status in frozen_layers.items():
        print(f"  {layer}: {'Frozen' if status else 'Active'}")

if __name__ == "__main__":
    main()
