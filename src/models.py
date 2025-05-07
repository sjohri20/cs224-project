import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from .utils import top_k_top_p_filtering

class RegressionHead(nn.Module):
    """
    A simple regression head that takes in an input vector and outputs a single scalar
    representing the predicted number of remaining tokens to be generated.
    The input dimension can be the hidden state dimension or hidden state plus one (if including token count).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        activated = self.relu(x)
        return self.linear(activated)

class ModelWithRegressorForLM(nn.Module):
    """
    Wraps a causal LM and attaches a regression head to each transformer block.
    At each autoregressive generation step, the model extracts the last-token hidden state
    from every block, optionally concatenates the current token count, and passes it through
    its corresponding regression head to predict the remaining number of tokens to be generated
    (rather than the final total output length).
    
    The language model is frozen (eval mode) so that only the regression heads are updated.
    """
    def __init__(self, model_name, tokenizer, include_token_count=False):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.include_token_count = include_token_count
        self.model_name = model_name
        self.tokenizer = tokenizer
        # Effective input dimension: add 1 if including token count.
        effective_dim = self.hidden_size + (1 if self.include_token_count else 0)
        self.regression_heads = nn.ModuleList([
            RegressionHead(effective_dim) for _ in range(self.num_layers)
        ])
        self.eos_token_id = self.model.config.eos_token_id
    def save_generation_step_metadata(self, metadata, step, hidden_states, output_dir):
        """
        metadata: dict with keys: 'datapoint_id', 'prompt_text', 'generated_text',
                  'tokens_generated', 'tokens_remaining'
        hidden_states: list of tensors (1 per layer) for a single datapoint
        """
        os.makedirs(output_dir, exist_ok=True)

        metadata_file = os.path.join(output_dir, "metadata.jsonl")
        # Save all layers' hidden states in one file
        hs_path = os.path.join(
            output_dir,
            f"hidden_step{step:04d}_{metadata['datapoint_id']}.npz"
        )
        # Convert list of tensors to numpy arrays and stack them
        hidden_states_np = np.stack([hs.cpu().numpy() for hs in hidden_states])
        np.savez(hs_path, hidden_states=hidden_states_np)
        
        with open(metadata_file, "a") as f:
                metadata["hidden_states_file"] = hs_path
                
                f.write(json.dumps(metadata) + "\n")
                
    def update_saved_metadata(self, unique_datapoint_id, generated_text, final_length, output_dir):
        """
        unique_datapoint_id: ID of the datapoint whose metadata is being updated
        total_steps: total number of steps in the final generated output
        """
        
        with open(os.path.join(output_dir, "metadata.jsonl"), "r") as f:
            content = f.read()
            metadata_list = [json.loads(line) for line in content.splitlines() if line.strip()]
            updated_lines = []
            for metadata in metadata_list:
                if metadata.get("datapoint_id") == unique_datapoint_id:
                    metadata["final_steps"] = final_length
                    metadata["generated_text"] = generated_text
                    updated_lines.append(json.dumps(metadata))
                else:
                    updated_lines.append(json.dumps(metadata))

        # Write back
        with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
            f.write("\n".join(updated_lines) + "\n")
    def generate_with_regression(self, input_ids, attention_mask, max_new_tokens=10,
                                 beam_width=1, top_k=0, top_p=0.0, unique_datapoint_id=None, prompt=None):
        """
        Autoregressively generates tokens while collecting regression outputs.
        At each step, each regression head receives the last token's hidden state,
        optionally concatenated with the current sequence length, and outputs a prediction
        of the remaining number of tokens to be generated (not the total sequence length).
        """
        device = input_ids.device

        if beam_width == 1:
            # Sampling mode.
            generated_ids = input_ids
            regression_outputs_per_step = []
            for step in range(max_new_tokens):
                outputs = self.model(input_ids=generated_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
                # For each transformer block (skip initial embeddings), extract last token's hidden state.
                last_token_hidden_states = [h[:, -1, :] for h in outputs.hidden_states[1:]]
                step_regression_outputs = {}
                for i, hidden_state in enumerate(last_token_hidden_states):
                    # Optionally include the current token count.
                    if self.include_token_count:
                        # Use the current sequence length as token count.
                        token_count = generated_ids.size(1)
                        token_count_tensor = torch.full((hidden_state.size(0), 1), 
                                                        fill_value=token_count, 
                                                        device=device, 
                                                        dtype=hidden_state.dtype)
                        input_to_regressor = torch.cat([hidden_state, token_count_tensor], dim=-1)
                    else:
                        input_to_regressor = hidden_state
                    metadata = {
                            "datapoint_id": unique_datapoint_id,
                            "prompt_text": prompt,
                            "model_name": self.model_name,
                            "steps_taken": step + 1,
                            "layer": i,
                            "top_k": top_k,
                            "top_p": top_p,
                            "beam_width": beam_width
                        }
                    if step % 5 == 0:
                        self.save_generation_step_metadata(metadata, step, last_token_hidden_states, "saved_embeddings")
                    # Predict remaining tokens (instead of total length)
                    reg_out = self.regression_heads[i](input_to_regressor)  # [batch_size, 1]
                    step_regression_outputs[f"layer_{i}"] = reg_out
                regression_outputs_per_step.append(step_regression_outputs)
                
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                # Filter logits for each example.
                filtered_logits = []
                for logit in logits:
                    filtered = top_k_top_p_filtering(logit.clone(), top_k=top_k, top_p=top_p)
                    filtered_logits.append(filtered.unsqueeze(0))
                filtered_logits = torch.cat(filtered_logits, dim=0)
                probs = F.softmax(filtered_logits, dim=-1)
                
                # Sample next token.
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.ones_like(generated_ids)
                
                # Early stopping if EOS is generated.
                if (next_token == self.eos_token_id).all():
                    # Decode the entire sequence when EOS is encountered
                    break
            print("Final step: ", step)
            self.update_saved_metadata(unique_datapoint_id,  self.tokenizer.decode(generated_ids[0]),step, "saved_embeddings")     
            return generated_ids, regression_outputs_per_step
        
        else:
            # Beam search mode.
            beams = [{
                "sequence": input_ids,
                "score": 0.0,
                "regression_outputs": []
            }]
            
            for step in range(max_new_tokens):
                all_candidates = []
                for beam in beams:
                    seq = beam["sequence"]
                    att_mask = torch.ones_like(seq)
                    outputs = self.model(input_ids=seq,
                                         attention_mask=att_mask,
                                         output_hidden_states=True)
                    last_token_hidden_states = [h[:, -1, :] for h in outputs.hidden_states[1:]]
                    step_regression_outputs = {}
                    for i, hidden_state in enumerate(last_token_hidden_states):
                        if self.include_token_count:
                            token_count = seq.size(1)
                            token_count_tensor = torch.full((hidden_state.size(0), 1),
                                                            fill_value=token_count,
                                                            device=device,
                                                            dtype=hidden_state.dtype)
                            input_to_regressor = torch.cat([hidden_state, token_count_tensor], dim=-1)
                        else:
                            input_to_regressor = hidden_state
                        # Predict remaining tokens (instead of total length)
                        reg_out = self.regression_heads[i](input_to_regressor)
                        step_regression_outputs[f"layer_{i}"] = reg_out
                        
                    logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab_size]
                    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    
                    num_candidates = top_k if top_k > 0 else probs.size(-1)
                    top_probs, top_indices = torch.topk(probs, num_candidates)
                    
                    for i in range(num_candidates):
                        token_id = top_indices[i].unsqueeze(0).unsqueeze(0)  # shape: [1, 1]
                        token_prob = top_probs[i].item()
                        new_seq = torch.cat([seq, token_id.to(device)], dim=1)
                        new_score = beam["score"] + torch.log(torch.tensor(token_prob + 1e-10)).item()
                        new_regression = beam["regression_outputs"] + [step_regression_outputs]
                        candidate = {
                            "sequence": new_seq,
                            "score": new_score,
                            "regression_outputs": new_regression
                        }
                        all_candidates.append(candidate)
                        
                all_candidates = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:beam_width]
                beams = all_candidates
                
                if all((beam["sequence"][0, -1].item() == self.eos_token_id) for beam in beams):
                    break
            
            best_beam = max(beams, key=lambda x: x["score"])
            return best_beam["sequence"], best_beam["regression_outputs"]

    def train_regressor_on_prompt(self, input_ids, attention_mask, max_new_tokens, optimizer,
                                  beam_width=1, top_k=0, top_p=0.0, unique_datapoint_id=None, prompt=None):
        """
        Generates an output given a prompt and computes regression loss.
        The target for every regression head at every generation step is the remaining number of tokens
        to be generated (not the total sequence length).
        Only the regression heads are updated while the LM remains frozen.
        If a regression head is frozen (requires_grad=False), it will be excluded from loss calculation.
        """
        generated_ids, regression_outputs = self.generate_with_regression(
            input_ids, attention_mask, max_new_tokens, beam_width, top_k, top_p, unique_datapoint_id, prompt,  
        )
        
        final_length = generated_ids.size(1)
        initial_length = input_ids.size(1)
        
        # Calculate loss per layer
        layer_losses = {}
        step_layer_losses = {}  # Store losses by step and layer for detailed analysis
        total_loss = 0.0
        count = 0
        
        # Track gradients for analysis
        gradient_data = {f"layer_{i}": [] for i in range(self.num_layers)}
        
        for step_idx, step_preds in enumerate(regression_outputs):
            # Current position in the sequence
            current_position = initial_length + step_idx
            # Remaining tokens to be generated
            remaining_tokens = final_length - current_position
            target = torch.tensor(remaining_tokens, dtype=torch.float, device=generated_ids.device)
            
            # Track step-specific losses
            if step_idx not in step_layer_losses:
                step_layer_losses[step_idx] = {}
            
            for layer, pred in step_preds.items():
                layer_idx = int(layer.split('_')[1])
                
                # Skip loss calculation for frozen layers
                if not any(p.requires_grad for p in self.regression_heads[layer_idx].parameters()):
                    layer_losses[layer] = 0.0  # Still record it but as zero
                    step_layer_losses[step_idx][layer] = 0.0
                    continue
                    
                # Calculate loss for this step and layer
                mse = F.mse_loss(pred.squeeze(-1), target.expand_as(pred.squeeze(-1)))
                
                # Store the individual losses before accumulating for backward
                step_layer_losses[step_idx][layer] = mse.item()
                
                # Track the gradient for this layer at this step
                if pred.requires_grad:
                    # Create a separate backward pass for gradient analysis
                    mse.backward(retain_graph=True)
                    
                    # Record gradient norm for each parameter in this head
                    param_grads = []
                    for param in self.regression_heads[layer_idx].parameters():
                        if param.grad is not None:
                            param_grads.append(param.grad.norm().item())
                    
                    gradient_data[layer].append({
                        'step': step_idx,
                        'position': current_position,
                        'target': remaining_tokens,
                        'prediction': pred.item(),
                        'loss': mse.item(),
                        'gradient_norm': sum(param_grads) / len(param_grads) if param_grads else 0
                    })
                    
                    # Zero out gradients after recording
                    for param in self.regression_heads[layer_idx].parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                
                # Accumulate loss for the actual optimization step
                total_loss += mse
                count += 1
                
                # Track loss per layer
                if layer not in layer_losses:
                    layer_losses[layer] = []
                if isinstance(layer_losses[layer], list):
                    layer_losses[layer].append(mse.item())
                else:
                    # Convert from direct value to list if needed
                    layer_losses[layer] = [mse.item()]
        
        # Average the losses
        avg_loss = total_loss / count if count > 0 else 0.0
        
        # Calculate average loss per layer
        avg_layer_losses = {}
        for layer, losses in layer_losses.items():
            if isinstance(losses, list) and losses:
                avg_layer_losses[layer] = sum(losses) / len(losses)
            else:
                avg_layer_losses[layer] = losses  # Already a direct value (0.0)
        
        # Only perform backward if there are active layers
        if count > 0:
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
        
        return generated_ids, regression_outputs, avg_loss, avg_layer_losses, step_layer_losses, gradient_data