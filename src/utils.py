import torch
import torch.nn.functional as F
import wandb
import json
from pathlib import Path
from datetime import datetime
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
            target = torch.tensor(final_length, dtype=torch.float, device=generated_ids.device)
            
            # Calculate losses without backpropagation
            step_losses = {}
            total_step_loss = 0.0
            count = 0
            
            for step_idx, step_preds in enumerate(regression_outputs):
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
    
    with torch.no_grad():  # No gradients needed for validation
        for idx, row in val_prompts.iterrows():
            prompt = row['prompt']
            print(f"Processing validation prompt {idx+1}/{len(val_prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
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
            target = torch.tensor(final_length, dtype=torch.float, device=generated_ids.device)
            
            # Calculate losses without backpropagation
            step_losses = {}
            total_step_loss = 0.0
            count = 0
            
            for step_idx, step_preds in enumerate(regression_outputs):
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
    
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    return avg_val_loss, avg_val_layer_losses, val_generations