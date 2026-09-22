import os
import torch
import sacrebleu
import re
import argparse

from sonnet_generation import SonnetGPT

def load_dev_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
    
    prompts = []
    references = []
    
    for sonnet in sonnets:
        lines = [line for line in sonnet.strip().split('\n') if line.strip()]
        if len(lines) >= 3:
            # prompt
            prompt = '\n'.join(lines[:3]) + '\n'
            # the remaining is ground truth
            reference = '\n'.join(lines[3:])
            
            prompts.append(prompt)
            references.append(reference)
            
    return prompts, references

@torch.no_grad()
def evaluate_models(model_paths, dev_file_path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("Loading Dev Data...")
    prompts, references = load_dev_data(dev_file_path)
    print(f"Found {len(prompts)} sonnets for evaluation.\n")

    results = {}

    for path in model_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"========== Evaluating {path} ==========")

        saved = torch.load(path, map_location=device, weights_only=False)
        model = SonnetGPT(saved['args'])
        model.load_state_dict(saved['model'])
        model = model.to(device)
        model.eval()

        hypotheses = []
        
        # generating poems
        for i, prompt in enumerate(prompts):
            # emcoding prompt
            encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)

            _, generated_output = model.generate(encoding['input_ids'], temperature=0.7, top_p=0.9)
            
            hypotheses.append(generated_output.strip())
            
        # calculating chrf
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        results[path] = chrf.score
        
        print(f"--> chrF Score: {chrf.score :.2f}\n")
    
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for path, score in sorted_results:
        print(f"Score: {score:.2f} | Model: {path}")


if __name__ == "__main__":
    my_models = [
        "batchsize4_lr3e-5_5_3.928.pt", 
        "batchsize8_lr3e-5_7_3.839.pt",
        "batchsz4_lr1e-5_10_3.888.pt",
        "batchsz8_lr5e-5_5_3.867.pt"
    ]
    
    dev_file = "data/TRUE_sonnets_held_out_dev.txt"
    
    evaluate_models(my_models, dev_file)