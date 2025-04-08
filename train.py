import argparse
import yaml
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
import wandb
import os
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from src.models import ModelWithRegressorForLM
from src.utils import run_validation_with_save, save_generation_to_file

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
    
    # Create directory for layer-specific models
    os.makedirs("layer_models", exist_ok=True)
    
    # Initialize tracking metrics table
    headers = ["Epoch", "Step", "LR"] + \
              [f"Train_L{i}" for i in range(model.num_layers)] + \
              [f"Val_L{i}" for i in range(model.num_layers)]
    print(" | ".join(headers))
    
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()  # Set model (regression heads) to training mode
        total_train_loss = 0.0
        train_layer_losses = {f"layer_{i}": 0.0 for i in range(model.num_layers)}
        samples_since_validation = 0
        
        for idx, row in train_prompts.iterrows():
            global_step += 1
            prompt = row['prompt']  # Assuming 'prompt' is the column name
            print(f"Processing training prompt {idx+1}/{len(train_prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            generated_ids, regression_outputs, loss, layer_losses = model.train_regressor_on_prompt(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                optimizer,
                beam_width=beam_width,
                top_k=top_k,
                top_p=top_p
            )
            
            generated_text = tokenizer.decode(generated_ids[0])
            final_length = generated_ids.size(1)
            total_train_loss += loss.item()
            
            # Save generated text to file
            generation_data = {
                "epoch": epoch + 1,
                "step": global_step,
                "prompt": prompt,
                "generated_text": generated_text,
                "token_count": final_length,
                "loss": loss.item(),
                "layer_losses": {k: v for k, v in layer_losses.items()},
                "timestamp": datetime.now().isoformat()
            }
            save_generation_to_file(train_outputs_file, generation_data)
            
            # Accumulate layer losses
            for layer, val in layer_losses.items():
                train_layer_losses[layer] += val
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"  Generated {final_length} tokens. Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            
            # Log predictions from the last step
            last_step_preds = {}
            if regression_outputs:
                last_preds = regression_outputs[-1]
                for layer, pred in last_preds.items():
                    last_step_preds[f"final_{layer}_pred"] = pred.item()
            
            # Format layer losses for logging
            log_layer_losses = {f"train_loss_{layer}": val for layer, val in layer_losses.items()}
            
            # Log learning rate
            wandb.log({
                "learning_rate": current_lr,
                "train_loss": loss.item(),
                "actual_length": final_length,
                **last_step_preds,
                **log_layer_losses,
                "epoch": epoch + 1,
                "step": global_step,
            })
            
            samples_since_validation += 1
            
            # Run validation at specified frequency if val_freq > 0
            if val_freq > 0 and samples_since_validation >= val_freq:
                print(f"\nRunning validation after {samples_since_validation} training samples...")
                samples_since_validation = 0
                
                # Current training stats
                curr_avg_train_loss = total_train_loss / (idx + 1)
                curr_avg_train_layer_losses = {layer: loss / (idx + 1) 
                                              for layer, loss in train_layer_losses.items()}
                
                # Run validation
                avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                    model, tokenizer, val_prompts, device, 
                    max_new_tokens, beam_width, top_k, top_p,
                    val_outputs_file, epoch, global_step
                )
                
                # Log metrics table row
                metrics_row = [f"{epoch+1}", f"{global_step}", f"{current_lr:.6f}"]
                for i in range(model.num_layers):
                    layer_key = f"layer_{i}"
                    metrics_row.append(f"{curr_avg_train_layer_losses.get(layer_key, 0):.4f}")
                for i in range(model.num_layers):
                    layer_key = f"layer_{i}"
                    metrics_row.append(f"{avg_val_layer_losses.get(layer_key, 0):.4f}")
                print(" | ".join(metrics_row))
                
                # Log mid-epoch stats
                wandb.log({
                    "epoch_avg_train_loss": curr_avg_train_loss,
                    "epoch_avg_val_loss": avg_val_loss,
                    **{f"epoch_avg_train_{layer}": loss for layer, loss in curr_avg_train_layer_losses.items()},
                    **{f"epoch_avg_val_{layer}": loss for layer, loss in avg_val_layer_losses.items()},
                    "epoch": epoch + 1,
                    "step": global_step,
                })
                
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
                    }, save_path)
                
                # Check for per-layer model improvement
                improved_layers = []
                layers_not_improved = []
                
                for layer, val_loss in avg_val_layer_losses.items():
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
                
                if improved_layers:
                    print(f"Improved layers in this step: {', '.join(improved_layers)}")
                if layers_not_improved:
                    print(f"Layers without improvement: {', '.join(layers_not_improved)}")
                
                # Check if any layer has exceeded early stopping threshold
                layers_to_stop = [layer for layer, counter in layer_early_stopping_counters.items() 
                                 if counter >= early_stopping]
                if layers_to_stop:
                    print(f"Early stopping triggered for layers: {', '.join(layers_to_stop)}")
                    print(f"Training stopped after {idx+1} samples in epoch {epoch+1}")
                    break
                
                # Resume training
                model.train()
            
            # Step the scheduler based on iteration
            if scheduler and scheduler_type == "cosine_warm_restarts":
                scheduler.step(epoch + idx / len(train_prompts))
        
        # Check if any layer has exceeded early stopping threshold
        layers_to_stop = [layer for layer, counter in layer_early_stopping_counters.items() 
                         if counter >= early_stopping]
        if layers_to_stop:
            print(f"Early stopping triggered for layers: {', '.join(layers_to_stop)}")
            break
            
        # Run validation at the end of each epoch regardless of val_freq
        if val_freq == 0 or samples_since_validation > 0:
            # Calculate average training losses for the full epoch
            avg_train_loss = total_train_loss / len(train_prompts)
            avg_train_layer_losses = {layer: loss / len(train_prompts) 
                                     for layer, loss in train_layer_losses.items()}
            
            print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            avg_val_loss, avg_val_layer_losses, val_generations = run_validation_with_save(
                model, tokenizer, val_prompts, device, 
                max_new_tokens, beam_width, top_k, top_p,
                val_outputs_file, epoch, global_step
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics table row
            metrics_row = [f"{epoch+1}", f"{global_step}", f"{current_lr:.6f}"]
            for i in range(model.num_layers):
                layer_key = f"layer_{i}"
                metrics_row.append(f"{avg_train_layer_losses.get(layer_key, 0):.4f}")
            for i in range(model.num_layers):
                layer_key = f"layer_{i}"
                metrics_row.append(f"{avg_val_layer_losses.get(layer_key, 0):.4f}")
            print(" | ".join(metrics_row))
            
            # Log epoch averages
            wandb.log({
                "epoch_avg_train_loss": avg_train_loss,
                "epoch_avg_val_loss": avg_val_loss,
                "learning_rate": current_lr,
                **{f"epoch_avg_train_{layer}": loss for layer, loss in avg_train_layer_losses.items()},
                **{f"epoch_avg_val_{layer}": loss for layer, loss in avg_val_layer_losses.items()},
                "epoch": epoch + 1,
                "step": global_step,
                "completed_epoch": True,
            })
            
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
                }, save_path)
            
            # Check for per-layer model improvement
            improved_layers = []
            layers_not_improved = []
            
            for layer, val_loss in avg_val_layer_losses.items():
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
            
            if improved_layers:
                print(f"Improved layers in this epoch: {', '.join(improved_layers)}")
            if layers_not_improved:
                print(f"Layers without improvement: {', '.join(layers_not_improved)}")
            
            # Check if any layer has exceeded early stopping threshold
            layers_to_stop = [layer for layer, counter in layer_early_stopping_counters.items() 
                             if counter >= early_stopping]
            if layers_to_stop:
                print(f"Early stopping triggered for layers: {', '.join(layers_to_stop)}")
                print(f"Training stopped after {epoch+1} epochs")
                break
        
        # Step the scheduler based on epoch
        if scheduler and scheduler_type == "cosine":
            scheduler.step()
    
    print(f"Training complete. Best overall validation loss: {best_overall_val_loss:.4f}")
    print("Best validation losses per layer:")
    for layer, loss in best_layer_val_losses.items():
        print(f"  {layer}: {loss:.4f}")

if __name__ == "__main__":
    main()
