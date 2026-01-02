import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Any
from utils import stable_softmax, stable_log_softmax, compute_entropy, js_divergence, compute_layer_confidence
from evolution import EnhancedDCLEDEvolutionEngine, JSLayerSelector, DynamicLayerSignalComputer
from config import get_model_adaptive_config, get_model_size_category
from logging_utils import get_logger
import math
import numpy as np
import json
import os
import time
import pandas as pd
import random
import gc
import re
logger = get_logger()

EPS = 1e-9
LOG_EPS = 1e-12
PROB_CLAMP_MIN = 1e-8
PROB_CLAMP_MAX = 1.0 - 1e-8
LOGIT_CLIP_MAX = 88.0
class UnifiedDCLED:
 
    def __init__(self, model_name: str, device: str = 'cuda',
                 num_gpus: str = '1', max_gpu_memory: int = 80):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stopping_criteria = None
        self.stop_words = []
     
        self.model, self.tokenizer = self._load_model(model_name)
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        self.model_size_category = get_model_size_category(model_name)
     
        logger.info(f"[Model] Loaded {model_name}")
        logger.info(f"[Model] {self.num_layers} layers, size category: {self.model_size_category}")
     
        device_obj = torch.device(device) if isinstance(device, str) else device
        self.js_selector = JSLayerSelector(device_obj)
        self.dcled_engine = None
        self.signal_computer = None
 
    def _load_model(self, model_name: str):
       
        if self.device == "cuda":
            kwargs = {
                "torch_dtype": torch.float16,
                "offload_folder": f"{model_name.replace('/', '_')}/offload"
            }
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(self.num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {
                            i: f"{self.max_gpu_memory}GiB"
                            for i in range(num_gpus)
                        },
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
     
        tokenizer_name = model_name
        if 'vicuna' in model_name.lower():
            tokenizer_name = 'huggyllama/llama-7b'
     
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
     
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            **kwargs
        )
     
        if self.device == "cuda" and self.num_gpus == "1":
            model.cuda()
     
        model.eval()
        return model, tokenizer
 
    def set_stop_words(self, stop_words: List[str]):
       
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
     
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
     
        self.stopping_criteria.append(LLaMAQAStoppingCriteria(list_stop_word_ids))
 
    def _get_model_device(self):
        if hasattr(self.model, 'device'):
            return self.model.device
        elif hasattr(self.model, 'parameters'):
            return next(self.model.parameters()).device
        else:
            return torch.device(self.device)
 
    def _get_lm_head_device(self):
        lm_head = self.model.lm_head
        if hasattr(lm_head, 'weight'):
            return lm_head.weight.device
        else:
            return next(lm_head.parameters()).device
         
    def _dola_score(
        self,
        dict_outputs: Dict[int, torch.Tensor],
        mature_layer: int,
        candidate_premature_layers: List[int],
        prefix_ids: torch.Tensor,
        input_ids: torch.Tensor,
        continue_ids: torch.Tensor,
        config: Dict,
        relative_top: float,
        relative_top_value: float,
        dola_alpha: float
    ) -> Tuple[float, Dict]:
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
        premature_layers = []
     
        available = [l for l in candidate_premature_layers if l in dict_outputs]
     
        if not available or mature_layer not in dict_outputs:
            out = stable_log_softmax(dict_outputs[mature_layer][0], dim=-1)
            out = out[prefix_ids.shape[-1] - 1: -1]
            return out[range(len(continue_ids)), continue_ids].mean().item(), {}
     
        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
            stacked_premature = torch.stack([
                dict_outputs[i][0, seq_i, :] for i in available
            ], dim=0)
         
            softmax_mature = stable_softmax(dict_outputs[mature_layer][0, seq_i, :], dim=-1)
            softmax_premature = stable_softmax(stacked_premature, dim=-1)
         
            js_divs = [js_divergence(softmax_mature, softmax_premature[i]).item()
                      for i in range(softmax_premature.shape[0])]
         
            selected_idx = int(np.argmax(js_divs)) if js_divs else 0
            selected_layer = available[selected_idx]
         
            premature_layer_dist[selected_layer] = premature_layer_dist.get(selected_layer, 0) + 1
            premature_layers.append(selected_layer)
     
        mature_logits_seq = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1, :]
     
        base_logits = torch.zeros_like(mature_logits_seq)
        for i, layer_idx in enumerate(premature_layers):
            base_logits[i] = dict_outputs[layer_idx][0, prefix_ids.shape[-1] - 1 + i, :]
     
        seq_len = len(continue_ids)
        effective_alpha = dola_alpha * min(1.0, seq_len / 3.0)
     
        diff_logits = mature_logits_seq - effective_alpha * base_logits
     
        if relative_top > 0.0:
            relative_top_mask = get_relative_top_filter(mature_logits_seq, relative_top)
            diff_logits = torch.where(
                relative_top_mask,
                torch.tensor(relative_top_value, device=diff_logits.device, dtype=diff_logits.dtype),
                diff_logits
            )
     
        diff_log_probs = stable_log_softmax(diff_logits, dim=-1)
     
        log_probs = diff_log_probs[range(diff_log_probs.shape[0]), continue_ids].mean().item()
     
        return log_probs, premature_layer_dist
 
    def lm_score(
        self,
        input_text1: str,
        input_text2: str,
        mode: str = 'DCLED',
        mature_layer: Optional[int] = None,
        candidate_premature_layers: Optional[List[int]] = None,
        relative_top: float = 0.1,
        relative_top_value: float = -1000.0,
        post_softmax: bool = True,
        evolution_rate: Optional[float] = None,
        evolution_scale: Optional[int] = None,
        evolution_lower_bound: Optional[float] = None,
        dataset_type: str = 'truthfulqa',
        max_seq_length: int = 4096,
        dola_alpha: float = 1.0,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[float, Optional[Dict]]:
        config = get_model_adaptive_config(self.model_name, dataset_type)
     
        if evolution_rate is not None:
            config['evolution_rate'] = evolution_rate
        if evolution_scale is not None:
            config['evolution_scale'] = evolution_scale
        if evolution_lower_bound is not None:
            config['evolution_lower_bound'] = evolution_lower_bound
     
        combined = input_text1 + input_text2
        tokens = self.tokenizer(combined, return_tensors="pt", truncation=False)
     
        if tokens.input_ids.shape[1] > max_seq_length:
            prefix_tokens = self.tokenizer(input_text1, return_tensors="pt", truncation=False)
            suffix_tokens = self.tokenizer(input_text2, return_tensors="pt", truncation=False)
            suffix_len = suffix_tokens.input_ids.shape[1]
            max_prefix_len = max_seq_length - suffix_len - 10
            if max_prefix_len > 0:
                prefix_tokens_truncated = prefix_tokens.input_ids[0, -max_prefix_len:]
                input_text1 = self.tokenizer.decode(prefix_tokens_truncated, skip_special_tokens=True)
     
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(
                input_text, return_tensors="pt"
            ).input_ids.to(self._get_model_device())
         
            prefix_ids = self.tokenizer(
                input_text1, return_tensors="pt"
            ).input_ids.to(self._get_model_device())
         
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
         
            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = stable_log_softmax(outputs, dim=-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].mean().item()
                return log_probs, None
         
            if candidate_premature_layers is None:
                if config.get('use_layer_range', False):
                    start = int(self.num_layers * config.get('layer_range_start_ratio', 0.0))
                    end = int(self.num_layers * config.get('layer_range_end_ratio', 1.0))
                    candidate_premature_layers = list(range(start, end))
                else:
                    candidate_premature_layers = list(range(self.num_layers))
         
            if mature_layer is None:
                mature_layer = self.num_layers
         
            early_exit_layers = candidate_premature_layers + [mature_layer]
         
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
         
            hidden_states = outputs.hidden_states
            lm_head = self.model.lm_head
            lm_head_device = self._get_lm_head_device()
         
            dict_outputs = {}
            for layer_idx in early_exit_layers:
                if layer_idx < len(hidden_states):
                    hidden = hidden_states[layer_idx].to(lm_head_device)
                    dict_outputs[layer_idx] = lm_head(hidden)
         
            if mode == 'dola':
                return self._dola_score(
                    dict_outputs=dict_outputs,
                    mature_layer=mature_layer,
                    candidate_premature_layers=candidate_premature_layers,
                    prefix_ids=prefix_ids,
                    input_ids=input_ids,
                    continue_ids=continue_ids,
                    config=config,
                    relative_top=relative_top,
                    relative_top_value=relative_top_value,
                    dola_alpha=dola_alpha
                )
                     
            elif mode in ['SLED', 'DCLED']:
                use_dc = mode == 'DCLED'
             
                op_T = config['op_T']
                evo_rate = config['evolution_rate']
                evo_scale = config['evolution_scale']
                evo_lower = config['evolution_lower_bound']
             
                new_output_logits = dict_outputs[mature_layer].clone()
                available_layers = [l for l in candidate_premature_layers if l in dict_outputs]
                             
                if self.dcled_engine is None:
                    self.dcled_engine = EnhancedDCLEDEvolutionEngine(config, lm_head_device)
             
                signal_computer = DynamicLayerSignalComputer(config, lm_head_device)
             
                use_range = config.get('use_layer_range', False)
                if use_range:
                    early_layers, middle_layers, late_layers = signal_computer.get_layer_groups(
                        len(available_layers),
                        use_range=True,
                        range_start_ratio=config.get('layer_range_start_ratio', 0.0),
                        range_end_ratio=config.get('layer_range_end_ratio', 1.0)
                    )
                else:
                    early_layers, middle_layers, late_layers = signal_computer.get_layer_groups(
                        len(available_layers)
                    )
             
                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    current_logits = dict_outputs[mature_layer][0, seq_i, :].clone()
                 
                    if config.get('use_generation_gate', True) and use_dc:
                        current_probs = stable_softmax(current_logits, dim=-1)
                        max_prob = current_probs.max().item()
                     
                        if max_prob >= config.get('gen_confidence_threshold', 0.88):
                            top_idx = current_probs.argmax()
                            boost = 0.3 + 0.2 * max_prob
                            current_logits[top_idx] += boost
                            new_output_logits[0, seq_i, :] = current_logits
                            continue
                 
                    topk_probs, topk_indices = torch.topk(
                        stable_softmax(current_logits, dim=-1),
                        min(evo_scale, current_logits.shape[-1])
                    )
                 
                    premature_logits_list = [
                        dict_outputs[l][0, seq_i, :] for l in available_layers
                    ]
                 
                    if not premature_logits_list:
                        new_output_logits[0, seq_i, :] = current_logits
                        continue
                 
                    layer_probs = [stable_softmax(l, dim=-1) for l in premature_logits_list]
                    layer_confidences = [compute_layer_confidence(p) for p in layer_probs]
                    layer_entropies = [compute_entropy(p).item() for p in layer_probs]
                 
                    layer_weights = None
                    if config.get('use_entropy_weighted_layers', True):
                        max_ent = max(layer_entropies) + EPS
                        layer_weights = [(max_ent - e) / max_ent for e in layer_entropies]
                        total_w = sum(layer_weights) + EPS
                        layer_weights = [w / total_w for w in layer_weights]
                 
                    proxy_gradients = self.dcled_engine.compute_proxy_gradients(
                        mature_logits=current_logits,
                        premature_logits_list=premature_logits_list,
                        topk_indices=topk_indices,
                        evolution_scale=evo_scale,
                        layer_weights=layer_weights
                    )
                 
                    if use_dc and len(available_layers) > 3:
                        P_N = stable_softmax(current_logits, dim=-1)
                     
                        signal_early = signal_computer.compute_group_signal(
                            early_layers, layer_probs, layer_confidences, P_N, layer_entropies
                        )
                        signal_middle = signal_computer.compute_group_signal(
                            middle_layers, layer_probs, layer_confidences, P_N, layer_entropies
                        )
                        signal_late = signal_computer.compute_group_signal(
                            late_layers, layer_probs, layer_confidences, P_N, layer_entropies
                        )
                     
                        w_early = config['layer_weights']['early']
                        w_middle = config['layer_weights']['middle']
                        w_late = config['layer_weights']['late']
                        total_w = w_early + w_middle + w_late + EPS
                     
                        final_target = (w_early * signal_early + w_middle * signal_middle + w_late * signal_late) / total_w
                        final_target = (1 - config['signal_strength']) * P_N + config['signal_strength'] * final_target
                        final_target = final_target.clamp(min=PROB_CLAMP_MIN)
                        final_target = final_target / final_target.sum()
                     
                        direct_grad = final_target - P_N
                        blend_ratio = 0.55
                        proxy_gradients = (1 - blend_ratio) * proxy_gradients + blend_ratio * direct_grad
                     
                        selected_layer, _ = self.js_selector.select_layer(
                            current_logits, premature_logits_list, available_layers,
                            temperature=config.get('layer_selection_temperature', 0.5)
                        )
                     
                        if selected_layer in available_layers:
                            sel_idx = available_layers.index(selected_layer)
                            premature_selected = premature_logits_list[sel_idx]
                         
                            entropy_N = compute_entropy(P_N).item()
                            vocab_size = P_N.numel()
                            max_ent = math.log(max(vocab_size, 2))
                            norm_entropy = entropy_N / max_ent if max_ent > 0 else 0.0
                         
                            adaptive_alpha = config['dola_alpha_base'] * (
                                1.0 + (config['dola_alpha_entropy_scale'] - 1.0) * norm_entropy
                            )
                            adaptive_alpha = float(np.clip(adaptive_alpha, 0.3, 2.0))
                         
                            contrastive_strength = config.get('contrastive_strength', 0.25)
                            contrastive_adjust = adaptive_alpha * stable_softmax(premature_selected, dim=-1)
                            proxy_gradients = proxy_gradients - contrastive_strength * contrastive_adjust
                 
                    hidden = new_output_logits[0, seq_i, :].clone()
                                
                    for t in range(op_T):
                        lr_t = evo_rate * (1 - t / op_T)
                        probs = stable_softmax(hidden, dim=-1)
                        gradient = probs + proxy_gradients
                        hidden = hidden - lr_t * gradient
                 
                    evolved = torch.full_like(hidden, evo_lower)
                    evolved[topk_indices] = hidden[topk_indices]
                    new_output_logits[0, seq_i, :] = evolved
             
                if post_softmax:
                    log_output = stable_log_softmax(new_output_logits[0], dim=-1)
                else:
                    log_output = new_output_logits[0]
             
                log_output = log_output[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = log_output[range(log_output.shape[0]), continue_ids].sum().item()
             
                return log_probs, None
         
            else:

                raise ValueError(f"Unknown mode: {mode}")



