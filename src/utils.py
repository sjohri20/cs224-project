import torch
import torch.nn.functional as F
import wandb
import json
import numpy as np
import csv
import os
from pathlib import Path
from datetime import datetime

def save_validation_data_to_csv(regression_outputs, initial_length, final_length, epoch, global_step, prompt_idx, csv_dir="logs/val_token_data"):
    """
    Save detailed validation metrics to CSV files.
    """
    # Create directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # Filename format: val_epoch_step_prompt.csv
    filename = f"{csv_dir}/val_epoch_{epoch}_step_{global_step}_prompt_{prompt_idx}.csv"
    
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
            'prompt_idx': prompt_idx,
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

def save_generation_to_file(file_path, data):
    """
    Save generation data to a JSONL file.
    Creates parent directories if they don't exist.
    Appends to file if it exists.
    """
    # Create parent directories if they don't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write data to file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filters logits using top-k and/or nucleus (top-p) filtering.
    logits: 1D tensor of logits for a single token.
    Returns the filtered logits.
    """
    assert logits.dim() == 1, "Expected 1D tensor for a single token distribution"
    
    if top_k > 0:
        topk = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, topk)[0][-1]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def run_validation(model, tokenizer, val_prompts, device, max_new_tokens, beam_width, top_k, top_p):
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    val_layer_losses = {f"layer_{i}": 0.0 for i in range(model.num_layers)}
    
    with torch.no_grad():  # No gradients needed for validation
        for idx, row in val_prompts.iterrows():
            prompt = row['prompt']
            print(f"Processing validation prompt {idx+1}/{len(val_prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            initial_length = inputs["input_ids"].size(1)
            
            # Generate without updating the model
            generated_ids, regression_outputs = model.generate_with_regression(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                beam_width=beam_width,
                top_k=top_k,
                top_p=top_p
            )
            
            final_length = generated_ids.size(1)
            
            # Calculate losses without backpropagation
            step_losses = {}
            total_step_loss = 0.0
            count = 0
            
            for step_idx, step_preds in enumerate(regression_outputs):
                # Calculate remaining tokens
                current_position = initial_length + step_idx
                remaining_tokens = final_length - current_position
                target = torch.tensor(remaining_tokens, dtype=torch.float, device=generated_ids.device)
                
                for layer, pred in step_preds.items():
                    mse = F.mse_loss(pred.squeeze(-1), target.expand_as(pred.squeeze(-1)))
                    if layer not in step_losses:
                        step_losses[layer] = []
                    step_losses[layer].append(mse.item())
                    total_step_loss += mse.item()
                    count += 1
            
            # Average the losses
            val_loss = total_step_loss / count if count > 0 else 0.0
            total_val_loss += val_loss
            
            # Calculate average loss per layer for this sample
            avg_step_layer_losses = {layer: sum(losses)/len(losses) 
                                     for layer, losses in step_losses.items()}
            
            # Accumulate layer losses
            for layer, val in avg_step_layer_losses.items():
                val_layer_losses[layer] += val
            
            print(f"  Generated {final_length} tokens. Val Loss: {val_loss:.4f}")
            
            # Log validation metrics
            log_val_layer_losses = {f"val_loss_{layer}": val 
                                   for layer, val in avg_step_layer_losses.items()}
            
            wandb.log({
                "val_loss": val_loss,
                "val_actual_length": final_length,
                **log_val_layer_losses,
            })
    
    # Calculate average validation losses
    avg_val_loss = total_val_loss / len(val_prompts)
    avg_val_layer_losses = {layer: loss / len(val_prompts) 
                           for layer, loss in val_layer_losses.items()}
    
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    return avg_val_loss, avg_val_layer_losses

def run_validation_with_save(model, tokenizer, val_prompts, device, max_new_tokens, 
                            beam_width, top_k, top_p, output_file, epoch, global_step,
                            csv_log_dir="logs/val_token_data", frozen_layers=None):
    """
    Runs validation and saves generated text to a file.
    Returns validation loss metrics and a list of generations.
    """
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    val_layer_losses = {f"layer_{i}": 0.0 for i in range(model.num_layers)}
    val_generations = []
    
    # Collect metrics across entire validation set before logging
    # Format: {layer_idx: {position_name: [errors]}}
    validation_metrics = {}
    total_length_sum = 0
    
    with torch.no_grad():  # No gradients needed for validation
        for idx, row in val_prompts.iterrows():
            prompt = row['prompt']
            print(f"Processing validation prompt {idx+1}/{len(val_prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            initial_length = inputs["input_ids"].size(1)
            
            # Generate without updating the model
            generated_ids, regression_outputs = model.generate_with_regression(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                beam_width=beam_width,
                top_k=top_k,
                top_p=top_p
            )
            
            generated_text = tokenizer.decode(generated_ids[0])
            final_length = generated_ids.size(1)
            total_length_sum += final_length
            
            # Collect metrics for this prompt instead of logging immediately
            num_steps = len(regression_outputs)
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
            for idx, (pos_idx, pos_name) in enumerate(zip(positions, position_names)):
                if pos_idx < len(regression_outputs):
                    step_preds = regression_outputs[pos_idx]
                    current_position = initial_length + pos_idx
                    remaining_tokens = final_length - current_position
                    
                    # Calculate error for each layer
                    for layer, pred in step_preds.items():
                        # Skip frozen layers
                        layer_idx = int(layer.split('_')[1])
                        if frozen_layers and frozen_layers.get(f"layer_{layer_idx}", False):
                            continue
                        
                        pred_value = pred.item() if isinstance(pred.item(), (int, float)) else pred.item()[0]
                        error = abs(pred_value - remaining_tokens)
                        
                        # Initialize nested dictionary if needed
                        if layer_idx not in validation_metrics:
                            validation_metrics[layer_idx] = {}
                        if pos_name not in validation_metrics[layer_idx]:
                            validation_metrics[layer_idx][pos_name] = []
                            
                        # Collect error for aggregation
                        validation_metrics[layer_idx][pos_name].append(error)
            
            # Save detailed data to CSV but don't log to wandb yet
            csv_filename = save_validation_data_to_csv(
                regression_outputs,
                initial_length,
                final_length,
                epoch + 1,
                global_step,
                idx,
                csv_dir=csv_log_dir
            )
            print(f"  Saved detailed validation token data to {csv_filename}")
            
            # Calculate losses without backpropagation
            step_losses = {}
            total_step_loss = 0.0
            count = 0
            
            for step_idx, step_preds in enumerate(regression_outputs):
                # Calculate remaining tokens
                current_position = initial_length + step_idx
                remaining_tokens = final_length - current_position
                target = torch.tensor(remaining_tokens, dtype=torch.float, device=generated_ids.device)
                
                for layer, pred in step_preds.items():
                    mse = F.mse_loss(pred.squeeze(-1), target.expand_as(pred.squeeze(-1)))
                    if layer not in step_losses:
                        step_losses[layer] = []
                    step_losses[layer].append(mse.item())
                    total_step_loss += mse.item()
                    count += 1
            
            # Average the losses
            val_loss = total_step_loss / count if count > 0 else 0.0
            total_val_loss += val_loss
            
            # Calculate average loss per layer for this sample
            avg_step_layer_losses = {layer: sum(losses)/len(losses) 
                                     for layer, losses in step_losses.items()}
            
            # Accumulate layer losses
            for layer, val in avg_step_layer_losses.items():
                val_layer_losses[layer] += val
            
            print(f"  Generated {final_length} tokens. Val Loss: {val_loss:.4f}")
            
            # Save generation to file
            generation_data = {
                "epoch": epoch + 1,
                "step": global_step,
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": final_length,
                "loss": val_loss,
                "layer_losses": {k: v for k, v in avg_step_layer_losses.items()},
                "timestamp": datetime.now().isoformat(),
                "csv_data_file": csv_filename
            }
            save_generation_to_file(output_file, generation_data)
            val_generations.append(generation_data)
            
            # Log validation metrics - REMOVED, will log aggregated metrics at the end
    
    # Calculate average validation losses
    avg_val_loss = total_val_loss / len(val_prompts)
    avg_val_layer_losses = {layer: loss / len(val_prompts) 
                           for layer, loss in val_layer_losses.items()}
    
    # Now log aggregated validation metrics across all prompts
    avg_length = total_length_sum / len(val_prompts) if val_prompts else 0
    
    metrics_to_log = {
        "total_generated_length": avg_length,
        "step": global_step,
        "epoch": epoch + 1,
    }
    
    # Add average metrics for each layer and position
    for layer_idx, positions in validation_metrics.items():
        for pos_name, errors in positions.items():
            if errors:  # Only log if we have data
                avg_error = sum(errors) / len(errors)
                metrics_to_log[f"val_layer_{layer_idx}_loss_{pos_name}"] = avg_error
    
    # Log to wandb
    wandb.log(metrics_to_log)
    
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    return avg_val_loss, avg_val_layer_losses, val_generations