from typing import Dict, List, Tuple  
from tqdm import tqdm
import numpy as np
import argparse
import logging
import pandas as pd
from model import UnifiedDCLED
from data_loaders import load_truthfulqa_dataset, load_sealqa_dataset, load_new_benchmarks
from utils import clear_cuda_memory
from config import get_model_adaptive_config
from logging_utils import get_logger
logger = get_logger()
logger = logging.getLogger(__name__)

def format_best(best_answer: str) -> str:
    return " " + best_answer.strip()
def split_multi_answer(ans: str, sep: str = ';') -> List[str]:
    if not ans or pd.isna(ans):
        return []
    answers = ans.strip().split(sep)
    return [" " + a.strip() for a in answers if a.strip()]
def build_prompt_and_answer(question: str, answer: str) -> Tuple[str, str]:
    prompt = f"Q: {question}\nA:"
    return prompt, answer
def MC_calcs(scores_true: List[float], scores_false: List[float],
             ref_true: List[str], ref_best: str) -> Dict[str, float]:
    if not scores_true or not scores_false:
        return {'MC1': 0.0, 'MC2': 0.0, 'MC3': 0.0}
 
    max_true = max(scores_true)
    max_false = max(scores_false)
    mc1 = 1.0 if max_true > max_false else 0.0
 
    all_scores = scores_true + scores_false
    max_score = max(all_scores)
    all_scores_exp = [np.exp(s - max_score) for s in all_scores]
    total_exp = sum(all_scores_exp)
    true_prob_mass = sum(all_scores_exp[:len(scores_true)]) / total_exp
    mc2 = true_prob_mass
 
    all_with_labels = [(s, True) for s in scores_true] + [(s, False) for s in scores_false]
    all_with_labels.sort(key=lambda x: x[0], reverse=True)
    mc3 = 1.0 if all_with_labels[0][1] else 0.0
 
    return {'MC1': mc1, 'MC2': mc2, 'MC3': mc3}
    pass

def evaluate_new_benchmark(llm: UnifiedDCLED, data: List[Dict],
                          name: str, mode: str, args) -> Dict:
    correct = 0
    total = 0
 
    generate_kwargs = {
        "mode": mode,
        "dataset_type": name,
        "relative_top": args.relative_top,
        "temperature": args.temperature,
        "max_seq_length": 4096,
        "dola_alpha": args.dola_alpha,
    }
 
    for item in tqdm(data, desc=f"{name}"):
        try:
            if name == 'hotpotqa':
                question = item.get('question', '')
                answer = item.get('answer', '')
                context = item.get('context', '')
             
                if not question or not answer:
                    continue
             
                if isinstance(context, str):
                    context_text = context[:4000]
                elif isinstance(context, dict):
                    context_text = str(context)[:4000]
                else:
                    context_text = ""
             
                prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
             
                s_correct, _ = llm.lm_score(prompt, " " + answer, **generate_kwargs)
                s_wrong, _ = llm.lm_score(prompt, " I don't know", **generate_kwargs)
             
                if s_correct > s_wrong:
                    correct += 1
                total += 1
             
            elif name in ['seal_0', 'seal_hard']:
                question = item.get('question', '')
                answer = item.get('answer', '')
                documents = item.get('documents', [])
             
                if not question or not answer:
                    continue
             
                if isinstance(documents, list):
                    context = "\n\n".join(str(doc) for doc in documents)
                else:
                    context = str(documents)
             
                context = context[:6000]
             
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
             
                s_correct, _ = llm.lm_score(prompt, " " + answer, **generate_kwargs)
                s_wrong, _ = llm.lm_score(prompt, " I don't know", **generate_kwargs)
             
                if s_correct > s_wrong:
                    correct += 1
                total += 1
             
        except Exception as e:
            logger.debug(f"Error in {name}: {e}")
            continue
     
        clear_cuda_memory()
 
    acc = correct / total if total > 0 else 0.0
    logger.info(f"[{name}] Ranking Accuracy: {acc:.4f} ({correct}/{total})")
    return {"ranking_accuracy": acc, "total": total}
   
def evaluate_truthfulqa(llm: UnifiedDCSLED, dataset: List[Dict],
                       mode: str, args: argparse.Namespace) -> Dict[str, float]:
    logger.info(f"\n[TruthfulQA] Evaluating with mode: {mode}")
 
    config = get_model_adaptive_config(llm.model_name, 'truthfulqa')
 
    if mode == 'VanillaGreedy':
        mature_layer = None
        candidate_premature_layers = None
    else:
        if args.early_exit_layers is None:
            if config.get('use_layer_range', False):
                start = int(llm.num_layers * config.get('layer_range_start_ratio', 0.0))
                end = int(llm.num_layers * config.get('layer_range_end_ratio', 1.0))
                early_exit_layers = list(range(start, end)) + [llm.num_layers]
            else:
                early_exit_layers = list(range(llm.num_layers))
            mature_layer = early_exit_layers[-1]
            candidate_premature_layers = early_exit_layers[:-1]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
            mature_layer = early_exit_layers[-1]
            candidate_premature_layers = early_exit_layers[:-1]
     
        logger.info(f" Mature layer: {mature_layer}, Premature: {len(candidate_premature_layers)} layers")
 
    generate_kwargs = dict(
        mode=mode,
        mature_layer=mature_layer,
        candidate_premature_layers=candidate_premature_layers,
        relative_top=args.relative_top,
        relative_top_value=args.relative_top_value,
        post_softmax=args.post_softmax,
        dataset_type='truthfulqa',
        dola_alpha=args.dola_alpha,
        temperature=args.temperature,
    )
 
    result_dict = {
        'total_mc1': 0.0,
        'total_mc2': 0.0,
        'total_mc3': 0.0,
        'n_questions': 0
    }
 
    num_samples = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
 
    for sample in tqdm(dataset[:num_samples], desc=f"TruthfulQA ({mode})"):
        ref_best = format_best(sample['answer_best'])
        ref_true = split_multi_answer(sample['answer_true'])
        ref_false = split_multi_answer(sample['answer_false'])
     
        if not ref_true or not ref_false:
            continue
     
        scores_true = []
        scores_false = []
     
        for temp_ans in ref_true:
            prompt, answer = build_prompt_and_answer(sample['question'], temp_ans)
            log_probs, _ = llm.lm_score(prompt, answer, **generate_kwargs)
            scores_true.append(log_probs)
     
        for temp_ans in ref_false:
            prompt, answer = build_prompt_and_answer(sample['question'], temp_ans)
            log_probs, _ = llm.lm_score(prompt, answer, **generate_kwargs)
            scores_false.append(log_probs)
     
        scores = MC_calcs(scores_true, scores_false, ref_true, ref_best)
     
        if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
            continue
     
        result_dict['total_mc1'] += scores['MC1']
        result_dict['total_mc2'] += scores['MC2']
        result_dict['total_mc3'] += scores['MC3']
        result_dict['n_questions'] += 1
 
    n = result_dict['n_questions']
    if n > 0:
        result_dict['total_mc1'] /= n
        result_dict['total_mc2'] /= n
        result_dict['total_mc3'] /= n
 
    logger.info(f"[TruthfulQA] Results: MC1 = {result_dict['total_mc1']:.4f}, "
                f"MC2 = {result_dict['total_mc2']:.4f}, MC3 = {result_dict['total_mc3']:.4f} (n={n})")
 
    return result_dict
   
def evaluate_sealqa(llm: UnifiedDCSLED, dataset: List[Dict],
                   mode: str, args: argparse.Namespace) -> Dict:
    logger.info(f"[SEAL-QA] Evaluating with mode: {mode}")
 
    total_score = 0.0
    total_samples = 0
    correct_ranking = 0
 
    generate_kwargs = dict(
        mode=mode,
        dataset_type='sealqa',
        relative_top=args.relative_top,
        temperature=args.temperature,
        max_seq_length=4096,
        dola_alpha=args.dola_alpha,
    )
 
    for sample in tqdm(dataset, desc="SEAL-QA"):
        question = sample['question']
        documents = sample.get('documents', [])
     
        if isinstance(documents, list):
            context = "\n\n".join(str(doc) for doc in documents)
        else:
            context = str(documents)
     
        context = context[:8000]
        gold_answer = sample['answer']
     
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
     
        try:
            gold_score, _ = llm.lm_score(prompt, " " + gold_answer, **generate_kwargs)
            wrong_score, _ = llm.lm_score(prompt, " I don't know", **generate_kwargs)
         
            if gold_score > wrong_score:
                correct_ranking += 1
         
            total_score += gold_score
            total_samples += 1
         
        except Exception as e:
            logger.debug(f"[SEAL-QA] Error: {e}")
            continue
     
        clear_cuda_memory()
 
    avg_score = total_score / total_samples if total_samples > 0 else 0.0
    ranking_acc = correct_ranking / total_samples if total_samples > 0 else 0.0
 
    logger.info(f"[SEAL-QA] Average Log-Prob: {avg_score:.4f}")
    logger.info(f"[SEAL-QA] Ranking Accuracy: {ranking_acc:.4f}")
 
    return {
        'avg_log_prob': avg_score,
        'ranking_accuracy': ranking_acc,
        'total': total_samples

    }
