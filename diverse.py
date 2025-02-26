from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from datasets import load_dataset
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from submodlib.functions.facilityLocation import FacilityLocationFunction
import time
from transformers import AutoTokenizer, AutoModel
from deita.selection.scorer import Llama_Scorer
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class QualityChecker:
    def __init__(self, model_name="hkust-nlp/deita-quality-scorer"):
        print(f"Initializing quality checker with model: {model_name}")
        self.scorer = Llama_Scorer(model_name, is_vllm=True)
    
    def get_quality_score(self, instruction: str, output: str) -> float:
        try:
            return self.scorer.infer_quality(instruction, output)
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            return 0.0

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sentence_embeddings(sentences, model_name, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    all_sentence_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
        batch_sentences = sentences[i:i + batch_size]
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, 
                                return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
        
        sentence_embeddings = mean_pooling(outputs[0], encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        all_sentence_embeddings.append(sentence_embeddings.cpu())

    all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)
    return all_sentence_embeddings.numpy()

def do_fla(X, number_all, number_select):
    start_time = time.time()
    Y = X
    obj = FacilityLocationFunction(n=number_all, mode="dense", data=Y, metric="euclidean")
    greedyList = obj.maximize(budget=number_select, optimizer='LazyGreedy', 
                            stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    idx_list = [tuple_i[0] for tuple_i in greedyList]
    print('FLA time used:', (time.time()-start_time)/60, '(min)')
    return idx_list

def get_quality_scores(data, quality_checker, batch_size=32):
    """Calculate quality scores for all examples in the dataset"""
    scores = []
    instructions = data['instruction']
    inputs = data['input']
    outputs = data['output']
    total = len(instructions)
    
    for i in tqdm(range(0, total, batch_size), desc="Calculating quality scores"):
        batch_end = min(i + batch_size, total)
        batch_scores = []
        
        for j in range(i, batch_end):
            instruction = instructions[j]
            input_text = inputs[j]
            output = outputs[j]
            
            # Handle cases where input is empty
            input_text = str(input_text) if pd.notna(input_text) else ""
            input_text = input_text if input_text != "nan" else ""
            
            # Combine instruction and input
            if input_text.strip():
                full_instruction = f"{instruction} {input_text}"
            else:
                full_instruction = instruction
                
            score = quality_checker.get_quality_score(full_instruction, output)
            batch_scores.append(score)
            
        scores.extend(batch_scores)
    
    return np.array(scores)

def clustering_with_quality_and_fla(file_path, output_file, samples_per_cluster=20, use_quality=False, top_n=10000):
    """
    Perform clustering on the top N highest quality examples
    
    Args:
        file_path: Path to the input JSON file
        output_file: Path to save the output JSON file
        top_n: Number of top quality samples to select for clustering
        samples_per_cluster: Number of samples to select from each cluster using FLA
    """
    print(f"Loading dataset from {file_path}")
    ds = load_dataset('json', data_files=file_path)
    data = pd.DataFrame(ds['train'])
    
    # Initialize quality checker
    print("Initializing quality checker...")
    quality_checker = QualityChecker()
    
    # Get quality scores for all examples
    print("Calculating quality scores...")
    quality_scores = get_quality_scores(ds['train'], quality_checker)
    
    # Select top N examples by quality score
    top_indices = np.argsort(quality_scores)[-top_n:]
    selected_data = data.iloc[top_indices].reset_index(drop=True)
    
    print(f"\nSelected top {top_n} examples with quality scores ranging from "
          f"{quality_scores[top_indices[-1]]:.3f} to {quality_scores[top_indices[0]]:.3f}")
    
    # Prepare data
    instructions = data['instruction'].tolist()
    inputs = data['input'].tolist()
    
    # Select only high-quality samples
    selected_instructions = [instructions[i] for i in top_indices]
    selected_inputs = [inputs[i] for i in top_indices]
    
    # Combine texts
    texts = []
    for inst, inp in zip(selected_instructions, selected_inputs):
        # Handle empty input cases
        input_text = str(inp) if pd.notna(inp) else ""
        # Remove 'nan' and empty string cases
        input_text = input_text if input_text != "nan" else ""
        # Combine text
        combined_text = inst.strip()
        if input_text.strip():
            combined_text += " " + input_text.strip()
        texts.append(combined_text)
    
    # Get embeddings
    print("Generating embeddings...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = get_sentence_embeddings(texts, model_name)

    # PCA reduction
    print("Performing PCA...")
    pca = PCA(n_components=0.95)
    reduced_embeddings = pca.fit_transform(embeddings)

    # KMeans clustering
    n = len(reduced_embeddings)
    k = int(np.sqrt(n / 2))
    print(f"Clustering into {k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeddings)

    print(f"Number of clusters: {k}")
    print(f"Cluster sizes: {np.bincount(clusters)}")

    # Apply FLA to each cluster
    final_selected_indices = []
    
    for cluster_id in tqdm(range(k), desc="Processing clusters"):
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        if len(cluster_indices) <= samples_per_cluster:
            final_selected_indices.extend(cluster_indices)
        else:
            cluster_embeddings = embeddings[cluster_indices]
            selected_indices = do_fla(cluster_embeddings, 
                                    len(cluster_embeddings), 
                                    samples_per_cluster)
            global_indices = cluster_indices[selected_indices]
            final_selected_indices.extend(global_indices)

    # Save results
    selected_data = selected_data.iloc[final_selected_indices].copy()
    selected_data['cluster'] = clusters[final_selected_indices]
    if use_quality:
        selected_data['quality_score'] = quality_scores[top_indices[final_selected_indices]]
    
    # Save to the specified output file
    selected_data.to_json(output_file, orient='records', indent=4)
    
    print(f"\nTotal samples selected: {len(final_selected_indices)}")
    print(f"Results saved to: {output_file}")
    
    # Print quality statistics only when using quality filtering
    if use_quality:
        final_scores = selected_data['quality_score']
        print(f"\nQuality score statistics of selected samples:")
        print(f"Mean: {final_scores.mean():.3f}")
        print(f"Median: {final_scores.median():.3f}")
        print(f"Min: {final_scores.min():.3f}")
        print(f"Max: {final_scores.max():.3f}")
    
    return selected_data

if __name__ == '__main__':
    file_path = './data/alpaca_data.json'
    
    # You can easily adjust these parameters
    USE_QUALITY = True  # Whether to use quality filtering
    TOP_N = 35000  # Number of high-quality samples to select when using quality filtering
    SAMPLES_PER_CLUSTER = 75  # Number of samples to select from each cluster
    
    # Dynamically generate output filename based on parameters
    if USE_QUALITY:
        output_filename = f'quality_top{TOP_N}_cluster{SAMPLES_PER_CLUSTER}_results.json'
    else:
        output_filename = f'cluster{SAMPLES_PER_CLUSTER}_results.json'
    
    results = clustering_with_quality_and_fla(
        file_path,
        output_file=output_filename,
        samples_per_cluster=SAMPLES_PER_CLUSTER,
        use_quality=USE_QUALITY,
        top_n=TOP_N
    )