import re
import os
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer


lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        # 'embed_tokens', 'lm_head',
    ],
}


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return '/home/cheam/fintech_project3/checkpoints/Llama-2-7b-chat-hf' # if from_remote else 'base_models/Llama-2-7b-chat-hf'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    
    dataset_names = [d for d in names.split(',')]
    dataset_list = []
    
    for name in dataset_names:
        rep = 1
        if not os.path.exists(name):
            rep = int(name.split('*')[1]) if '*' in name else 1
            name = ('FinGPT/fingpt-forecaster-' if from_remote else '/home/cheam/fintech_project3/FinGPT-master/fingpt/FinGPT_Strategy/') + name.split('*')[0]
        tmp_dataset = datasets.load_dataset(name) if from_remote else datasets.load_from_disk(name)
    
        if 'test' not in tmp_dataset:
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)   
        dataset_list.extend([tmp_dataset] * rep)
    
    return dataset_list


def parse_answer(answer):
    
    portfolio_weights = re.search(r'\[Portfolio Weights\]:(.*?)\[Explanation\]', answer, re.DOTALL)
    explanation = re.search(r'\[Explanation\](.*?)\[Risk Management\]', answer, re.DOTALL)
    risk_management = re.search(r'\[Risk Management\](.*)', answer, re.DOTALL)
    if not portfolio_weights or not explanation or not risk_management:
        return None
    
    #pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(4)
    portfolio_weights, explanation, risk_management = portfolio_weights.group(1), explanation.group(1), risk_management.group(1)
        
        
    return {
        "portfolio weights": portfolio_weights,
        "explanation of reason": explanation,
        "risk management": risk_management
    }
    

def calc_rouge_score(references, answers):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    # if not answers_dict['prediction']:
    #     return {}
    
    # bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    # mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['portfolio weights'], answers_dict['portfolio weights'])
    cons_rouge_scores = calc_rouge_score(gts_dict['explanation of reason'], answers_dict['explanation of reason'])
    anal_rouge_scores = calc_rouge_score(gts_dict['risk management'], answers_dict['risk management'])
                              
    # print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Portfolio Weights: {pros_rouge_scores}")
    print(f"\nRouge Score of Explanation: {cons_rouge_scores}")
    print(f"\nRouge Score of Risk Management: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['portfolio weights']),
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
    