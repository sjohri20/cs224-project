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

    train_prompts = pd.read_csv(train_prompt_file)
    val_prompts = pd.read_csv(val_prompt_file)
    print(f"Loaded {len(train_prompts)} training prompts from {train_prompt_file}")
    print(f"Loaded {len(val_prompts)} validation prompts from {val_prompt_file}")
    
    # Freeze LM parameters.
    for param in model.model.parameters():
        param.requires_grad = False
    model.model.eval()
    print("Language model frozen, only regression heads will be trained")
    
    optimizer = torch.optim.Adam(model.regression_heads.parameters(), lr=learning_rate)
    
    # Set up the scheduler
    total_steps = len(train_prompts) * num_epochs
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
            
        for idx, row in train_prompts.iterrows():
            global_step += 1
            prompt = row['prompt']  # Assuming 'prompt' is the column name
            print(f"Processing training prompt {idx+1}/{len(train_prompts)}: {prompt[:50]}...")
            
            # Generate unique ID for the datapoint
            unique_id = f"train_{idx}"
            
            # Before training, freeze gradient flow for frozen layers
            for layer_idx in range(model.num_layers):
                layer_name = f"layer_{layer_idx}"
                if frozen_layers[layer_name]:
                    for param in model.regression_heads[layer_idx].parameters():
                        param.requires_grad = False
                else:
                    for param in model.regression_heads[layer_idx].parameters():
                        param.requires_grad = True
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            initial_length = inputs["input_ids"].size(1)
            generated_ids, regression_outputs, loss, layer_losses, step_layer_losses, gradient_data = model.train_regressor_on_prompt(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                optimizer,
                beam_width=beam_width,
                top_k=top_k,
                top_p=top_p,
                unique_datapoint_id=unique_id,
                prompt=prompt
            )
            
            generated_text = tokenizer.decode(generated_ids[0])
            print("generated text: ", generated_text)
            final_length = generated_ids.size(1)
            total_train_loss += loss.item()
            
            # Log simplified metrics to wandb (beginning and midpoint only)
            log_simplified_metrics_to_wandb(
                regression_outputs, 
                initial_length, 
                final_length,
                global_step,
                epoch + 1,
                frozen_layers
            )
            
            # Save detailed per-token data to CSV
            csv_filename = save_step_data_to_csv(
                regression_outputs,
                initial_length,
                final_length,
                global_step,
                epoch + 1,
                idx,
                csv_dir=csv_log_dir
            )
            print(f"Saved detailed token data to {csv_filename}")
            
            # Save generated text to file
            generation_data = {
                "epoch": epoch + 1,
                "step": global_step,
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": final_length,
                "loss": loss.item(),
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
            print(f"  Generated {final_length} tokens. Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            
            # Log only learning rate to wandb
            wandb.log({
                "lr": current_lr,
                "step": global_step,
                "epoch": epoch + 1
            })
            
            samples_since_validation += 1
            
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
                        samples_active = idx + 1
                        avg_loss = loss_sum / samples_active
                        curr_avg_train_layer_losses[layer] = avg_loss
                        curr_avg_train_loss += avg_loss
                        count_active += 1
                
                if count_active > 0:
                    curr_avg_train_loss /= count_active
                
                # Run validation
                avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                    model, tokenizer, val_prompts, device, 
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
                    print(f"All regression heads are frozen. Training stopped after {idx+1} samples in epoch {epoch+1}")
                    break
                
                # Resume training
                model.train()
            
            # Step the scheduler based on iteration
            if scheduler and scheduler_type == "cosine_warm_restarts":
                scheduler.step(epoch + idx / len(train_prompts))
        
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
                    samples_active = len(train_prompts)
                    avg_loss = loss_sum / samples_active
                    avg_train_layer_losses[layer] = avg_loss
                    avg_train_loss += avg_loss
                    count_active += 1
            
            if count_active > 0:
                avg_train_loss /= count_active
            
            print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                model, tokenizer, val_prompts, device, 
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
