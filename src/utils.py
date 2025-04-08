import torch
import torch.nn.functional as F
import wandb
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def log_validation_step_metrics(regression_outputs, initial_length, final_length, global_step, epoch, prompt_idx):
    """
    Logs detailed per-step and per-layer metrics for validation data.
    Similar to the training metrics but with validation-specific naming.
    """
    num_layers = len([k for k in regression_outputs[0].keys() if k.startswith("layer_")])
    num_steps = len(regression_outputs)
    
    # Create arrays to store step-wise values for each layer
    step_positions = []  # Token positions in the sequence
    step_targets = []    # Target values (remaining tokens) at each step
    layer_step_preds = {f"layer_{i}": [] for i in range(num_layers)}
    layer_step_errors = {f"layer_{i}": [] for i in range(num_layers)}
    
    for step_idx, step_preds in enumerate(regression_outputs):
        # Current position in the sequence
        current_position = initial_length + step_idx
        # Remaining tokens to be generated (target value)
        remaining_tokens = final_length - current_position
        
        step_positions.append(current_position)
        step_targets.append(remaining_tokens)
        
        for layer, pred in step_preds.items():
            # Record prediction
            pred_value = pred.item() if isinstance(pred.item(), (int, float)) else pred.item()[0]
            layer_step_preds[layer].append(pred_value)
            
            # Calculate and record absolute error
            abs_error = abs(pred_value - remaining_tokens)
            layer_step_errors[layer].append(abs_error)
    
    # Calculate average error by layer
    layer_avg_errors = {layer: sum(errors)/len(errors) if errors else 0 
                       for layer, errors in layer_step_errors.items()}
    
    # Log validation heatmap
    error_matrix = np.zeros((num_layers, num_steps))
    for i in range(num_layers):
        layer = f"layer_{i}"
        for j in range(min(num_steps, len(layer_step_errors[layer]))):
            error_matrix[i, j] = layer_step_errors[layer][j]
    
    wandb.log({
        f"val_prompt_{prompt_idx}_error_heatmap": wandb.plots.HeatMap(
            x_labels=[f"Pos {p}" for p in step_positions],
            y_labels=[f"Layer {i}" for i in range(num_layers)],
            matrix_values=error_matrix.tolist(),
            show_text=False
        ),
        "step": global_step,
        "epoch": epoch,
    })
    
    # Log layer comparison
    wandb.log({
        f"val_prompt_{prompt_idx}_layer_comparison": wandb.Table(
            columns=["Layer", "Average Error"],
            data=[[layer, error] for layer, error in layer_avg_errors.items()]
        ),
        "step": global_step,
        "epoch": epoch,
    })
    
    # Return the error metrics for aggregation
    return layer_avg_errors, step_positions, step_targets, layer_step_errors

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
                            beam_width, top_k, top_p, output_file, epoch, global_step):
    """
    Runs validation and saves generated text to a file.
    Returns validation loss metrics and a list of generations.
    """
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    val_layer_losses = {f"layer_{i}": 0.0 for i in range(model.num_layers)}
    val_generations = []
    
    # Collect per-position errors across all validation prompts
    all_position_errors = {}  # {position: [errors]}
    all_layer_errors = {f"layer_{i}": [] for i in range(model.num_layers)}
    
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
            
            # Log detailed step metrics for this validation prompt
            layer_avg_errors, step_positions, step_targets, layer_step_errors = log_validation_step_metrics(
                regression_outputs,
                initial_length,
                final_length,
                global_step,
                epoch + 1,
                idx
            )
            
            # Collect position errors for global analysis
            for pos_idx, pos in enumerate(step_positions):
                if pos not in all_position_errors:
                    all_position_errors[pos] = []
                
                # Average error across all layers at this position
                pos_errors = []
                for layer in layer_step_errors:
                    if pos_idx < len(layer_step_errors[layer]):
                        pos_errors.append(layer_step_errors[layer][pos_idx])
                
                if pos_errors:
                    all_position_errors[pos].append(sum(pos_errors) / len(pos_errors))
            
            # Collect layer errors for global analysis
            for layer, errors in layer_avg_errors.items():
                all_layer_errors[layer].append(errors)
            
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
                "timestamp": datetime.now().isoformat()
            }
            save_generation_to_file(output_file, generation_data)
            val_generations.append(generation_data)
            
            # Log validation metrics
            log_val_layer_losses = {f"val_loss_{layer}": val 
                                   for layer, val in avg_step_layer_losses.items()}
            
            wandb.log({
                "val_loss": val_loss,
                "val_actual_length": final_length,
                **log_val_layer_losses,
                "step": global_step,
            })
    
    # Calculate average validation losses
    avg_val_loss = total_val_loss / len(val_prompts)
    avg_val_layer_losses = {layer: loss / len(val_prompts) 
                           for layer, loss in val_layer_losses.items()}
    
    # Log aggregated validation metrics across all prompts
    
    # 1. Average error by position
    position_avg_errors = {pos: sum(errors)/len(errors) for pos, errors in all_position_errors.items() if errors}
    position_data = [[pos, avg_error] for pos, avg_error in sorted(position_avg_errors.items())]
    
    wandb.log({
        "val_error_by_position": wandb.Table(
            columns=["Position", "Average Error"],
            data=position_data
        ),
        "val_position_error_chart": wandb.plot.line(
            table=wandb.Table(
                columns=["Position", "Error"],
                data=position_data
            ),
            x="Position",
            y="Error",
            title="Validation Error by Token Position"
        ),
        "step": global_step,
        "epoch": epoch + 1,
    })
    
    # 2. Compare layer performance
    layer_avg_errors = {layer: sum(errors)/len(errors) if errors else 0 
                       for layer, errors in all_layer_errors.items()}
    layer_data = [[layer, avg_error] for layer, avg_error in layer_avg_errors.items()]
    
    wandb.log({
        "val_error_by_layer": wandb.Table(
            columns=["Layer", "Average Error"],
            data=layer_data
        ),
        "val_layer_error_chart": wandb.plot.bar(
            table=wandb.Table(
                columns=["Layer", "Error"],
                data=layer_data
            ),
            x="Layer",
            y="Error",
            title="Average Validation Error by Layer"
        ),
        "step": global_step,
        "epoch": epoch + 1,
    })
    
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    return avg_val_loss, avg_val_layer_losses, val_generations