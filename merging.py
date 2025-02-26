import re
import openai
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from deita.selection.scorer import Llama_Scorer
import json

name = "response_quality_top35000_cluster75_results.json"
# 文件名集中管理
FILE_NAMES = {
    "input": name,  # 输入文件名
    "output": "merged_" + name,  # 主输出文件名
    "fixed_output": name+"fixed_data.json",  # 修复后的 JSON 文件名
    "detailed_output": "merged_quality_"+name+"_detailed_results.json",  # 详细合并结果文件名
}

base_url = ""
api_key = ""

client = AsyncOpenAI(base_url=base_url, api_key=api_key)

MAX_CONCURRENT = 10
model = "gpt-4o"




# Example conversation history showing successful merges
from examples import *

class ContrieverMatcher:
    def __init__(self, model_name="fangyuan/nq_extractive_compressor"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_similarity_scores(self, texts: List[str]) -> np.ndarray:
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        similarity_matrix = torch.mm(all_embeddings, all_embeddings.transpose(0, 1)).numpy()
        return similarity_matrix

def find_similar_groups_in_cluster(cluster_data: List[Dict], matcher: ContrieverMatcher, 
                                 similarity_threshold: float = 0.8, 
                                 min_group_size: int = 2,
                                 max_group_size: int = 5) -> List[List[int]]:
    if len(cluster_data) < min_group_size:
        return []

    texts = []
    for item in cluster_data:
        instruction = item['instruction']
        input_text = item.get('input', '')
        if input_text and not isinstance(input_text, float):
            text = f"{instruction} {input_text}"
        else:
            text = instruction
        texts.append(text)
    
    similarity_matrix = matcher.get_similarity_scores(texts)
    groups = []
    used_indices = set()
    
    for i in range(len(texts)):
        if i in used_indices:
            continue
            
        similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
        similar_indices = similar_indices.tolist()
        similar_indices = [idx for idx in similar_indices if idx not in used_indices]
        
        if len(similar_indices) >= min_group_size:
            similar_indices = similar_indices[:max_group_size]
            groups.append(similar_indices)
            used_indices.update(similar_indices)
    
    return groups

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_chat_completion(message: str, semaphore) -> Dict:
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": '''You are a helpful assistant that creates high-quality instruction-following training examples by merging similar examples.
                     NOTED THAT DONT USE ANY "OR" to connect the logic. 
                     It is important to keep the instruction with Helpfulness, Relevance, Accuracy, Level of Details, and Structure. 
                     Always output in valid JSON format.'''},
                    {"role": "user", "content": message}
                ],
                temperature=0.25,
                timeout=80
            )
            response_result = response.choices[0].message.content.strip()
            response_result = response_result.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(response_result)
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
                if "instruction" not in result or "output" not in result:
                    raise ValueError("Missing required fields in response")
                if "input" in result and not isinstance(result["input"], str):
                    raise ValueError("Input field is not a string")
                if not isinstance(result["output"], str):
                    result['output'] = str(result['output'])
                    print('output field is not a String. So converted to string.')
                return result
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response content: {response_result}")
                raise
            except ValueError as e:
                print(f"Invalid response format: {e}")
                print(f"Response content: {response_result}")
                raise
    except Exception as e:
        print(f"Error in get_chat_completion: {type(e).__name__} - {str(e)}")
        raise

class QualityChecker:
    def __init__(self):
        model_name_or_path = "hkust-nlp/deita-quality-scorer"
        self.scorer = Llama_Scorer(model_name_or_path, is_vllm=True)
    
    def get_quality_score(self, instruction: str, output: str) -> float:
        try:
            return self.scorer.infer_quality(instruction, output)
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            return 0.0




async def merge_instructions_with_gpt4_async(data: List[Dict], similar_groups: List[List[int]], 
                                           quality_checker: QualityChecker = None) -> Tuple[List[Dict], List[List[int]], List[Dict]]:
    prompts = []
    original_scores = []
    merge_metadata = []
    successful_merges = []
    failed_merges = []
    
    for group in similar_groups:
        group_data = []
        
        # Start with the example conversation history
        messages = EXAMPLE_MESSAGES.copy()
        
        # Add the new examples to merge
        examples_text = "Please merge these examples:\n\n"
        
        for idx in group:
            item = data[idx]
            group_data.append({
                "original_index": idx,
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
                "cluster": item.get("cluster", None)
            })
            
            examples_text += f"Example {len(group_data)}:\n"
            examples_text += json.dumps({
                "instruction": item['instruction'],
                "input": item.get('input', '') if item.get('input', '') and not isinstance(item.get('input', ''), float) else "",
                "output": item['output']
            }, ensure_ascii=False, indent=2)
            examples_text += "\n\n"
        
        # Add the new examples as a user message
        messages.append({
            "role": "user",
            "content": examples_text
        })

        if quality_checker is not None:
            group_scores = []
            for item in group_data:
                score = quality_checker.get_quality_score(item['instruction'], item['output'])
                group_scores.append(score)
            avg_score = sum(group_scores) / len(group_scores)
            original_scores.append(avg_score)
            merge_metadata.append({
                "original_indices": group,
                "original_scores": group_scores,
                "average_score": avg_score
            })
        else:
            merge_metadata.append({
                "original_indices": group,
                "original_scores": [],
                "average_score": 0.0
            })
        
        prompts.append({"messages": messages, "original_data": group_data})

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    detailed_results = []
    
    for idx, prompt_data in enumerate(prompts):
        try:
            merged_result = await get_chat_completion(prompt_data["messages"], semaphore)
            if merged_result:
                source_indices = [d["original_index"] for d in prompt_data["original_data"]]
                source_instructions = [data[idx]["instruction"] for idx in source_indices]
                
                merged_result["source_data"] = {
                    "indices": source_indices,
                    "original_instructions": source_instructions,
                    "merge_group_id": f"group_{idx}"
                }
                
                detailed_result = {
                    "merge_group_id": f"group_{idx}",
                    "merged_result": merged_result,
                    "original_data": prompt_data["original_data"],
                    "source_indices": source_indices
                }

                if quality_checker is not None:
                    merged_score = quality_checker.get_quality_score(
                        merged_result['instruction'],
                        merged_result['output']
                    )
                    avg_original_score = merge_metadata[idx]["average_score"]
                    
                    detailed_result["metrics"] = {
                        "merged_quality_score": merged_score,
                        "original_scores": merge_metadata[idx]["original_scores"],
                        "average_original_score": avg_original_score,
                        "quality_improvement": merged_score - avg_original_score
                    }
                    
                    max_retries = 3
                    retry_count = 0
                    best_score = merged_score
                    best_result = merged_result
                    
                    while retry_count < max_retries and best_score <= avg_original_score:
                        print(f"Group {idx}: Retry {retry_count + 1}/{max_retries} "
                              f"(Current quality: {best_score:.2f} <= {avg_original_score:.2f})")
                        
                        retry_result = await get_chat_completion(prompt_data["messages"], semaphore)
                        if retry_result:
                            current_score = quality_checker.get_quality_score(
                                retry_result['instruction'],
                                retry_result['output']
                            )
                            if current_score > best_score:
                                best_result = retry_result
                                best_score = current_score
                        retry_count += 1
                    
                    if best_score > avg_original_score:
                        detailed_result["merge_status"] = "success"
                        merged_result = best_result
                        successful_merges.append((best_result, prompt_data["original_data"]))
                        print(f"Group {idx}: Merged successfully (Quality improvement: {best_score:.2f} > {avg_original_score:.2f})")
                    else:
                        detailed_result["merge_status"] = "failed_quality_check"
                        failed_merges.extend([idx for idx in similar_groups[idx]])
                        print(f"Group {idx}: Failed quality check")
                else:
                    detailed_result["metrics"] = {}
                    detailed_result["merge_status"] = "success"
                    successful_merges.append((merged_result, prompt_data["original_data"]))
                    print(f"Group {idx}: Merged successfully (Quality check disabled)")

                detailed_results.append(detailed_result)
                
        except Exception as e:
            print(f"Task failed: {e}")
            error_record = {
                "merge_status": "failed_error",
                "error": str(e),
                "original_data": prompt_data["original_data"]
            }
            detailed_results.append(error_record)
            failed_merges.extend([d["original_index"] for d in prompt_data["original_data"]])
    
    merged_results = [item[0] for item in successful_merges]
    original_indices_to_keep = failed_merges
    
    return merged_results, original_indices_to_keep, detailed_results
async def main_async(enable_quality_check: bool = True, test_mode: bool = False):
    quality_checker = QualityChecker() if enable_quality_check else None
    input_path = FILE_NAMES["input"]
    print(f"Reading data from {input_path}")
    print(f"Quality checking is {'enabled' if enable_quality_check else 'disabled'}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        initial_results = json.load(f)
    
    if test_mode:
        initial_results = initial_results[:10]  # 仅处理前10个样本
        print("Test mode enabled: Only processing first 10 samples.")
    
    initial_count = len(initial_results)
    print(f"Original data count: {initial_count}")
    
    cluster_groups = defaultdict(list)
    for idx, item in enumerate(initial_results):
        cluster_groups[item['cluster']].append(item)
    
    print(f"Found {len(cluster_groups)} clusters")
    matcher = ContrieverMatcher()
    
    all_similar_groups = []
    used_indices = set()
    
    for cluster_id, cluster_data in tqdm(cluster_groups.items(), desc="Finding similar groups"):
        print(f"\nProcessing cluster {cluster_id} with {len(cluster_data)} items")
        
        cluster_base_idx = sum(len(cluster_groups[cid]) for cid in cluster_groups if cid < cluster_id)
        similar_groups = find_similar_groups_in_cluster(
            cluster_data,
            matcher,
            similarity_threshold=0.55,
            min_group_size=2,
            max_group_size=2
        )
        
        print(f"Found {len(similar_groups)} similar groups in cluster {cluster_id}")
        
        for group in similar_groups:
            global_indices = [cluster_base_idx + idx for idx in group]
            all_similar_groups.append(global_indices)
            used_indices.update(global_indices)
    
    merged_results, keep_original_groups, detailed_results = await merge_instructions_with_gpt4_async(
        initial_results,
        all_similar_groups,
        quality_checker
    )
    
    final_results = []
    
    with open(FILE_NAMES["detailed_output"], 'w', encoding='utf-8') as f:
        json.dump({
            "merge_details": detailed_results,
            "statistics": {
                "total_original_data": initial_count,
                "total_groups_attempted": len(all_similar_groups),
                "successful_merges": len([r for r in detailed_results if r.get("merge_status") == "success"]),
                "failed_quality_check": len([r for r in detailed_results if r.get("merge_status") == "failed_quality_check"]),
                "failed_error": len([r for r in detailed_results if r.get("merge_status") == "failed_error"])
            },
            "metadata": {
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "similarity_threshold": 0.55,
                "min_group_size": 2,
                "max_group_size": 2,
                "quality_check_enabled": enable_quality_check
            }
        }, f, indent=2, ensure_ascii=False)
    
    if merged_results:
        final_results.extend(merged_results)
        print(f"\nSuccessfully merged {len(merged_results)} groups")
    
    for group in keep_original_groups:
        item = initial_results[group]
        final_item = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }
        final_results.append(final_item)
    
    unmerged_count = 0
    for idx, item in enumerate(initial_results):
        if idx not in used_indices:
            final_item = {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"]
            }
            final_results.append(final_item)
            unmerged_count += 1
    
    # 提取并保存修复后的 JSON 文件
    fixed_results = [{
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"]
    } for item in final_results]
    
    with open(FILE_NAMES["fixed_output"], 'w', encoding='utf-8') as f:
        json.dump(fixed_results, f, indent=2, ensure_ascii=False)
    
    print("\nData Statistics:")
    print(f"Original data count: {initial_count}")
    print(f"Merged data count: {len(used_indices)}")
    print(f"New data generated from merges: {len(merged_results) if merged_results else 0}")
    print(f"Unmerged original data: {unmerged_count}")
    print(f"Final data count: {len(final_results)}")
    print(f"Results saved to: {FILE_NAMES['output']}")
    print(f"Fixed results saved to: {FILE_NAMES['fixed_output']}")
    print(f"Detailed merge results saved to: {FILE_NAMES['detailed_output']}")

if __name__ == "__main__":
    start_time = time.time()
    print("Starting data processing...")
    
    def print_time_cost():
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\nTotal time: {hours} hours {minutes} minutes {seconds} seconds")
    
    enable_quality_check = True  # Set to False to disable quality checking
    test_mode = False  # 启用测试模式，仅处理前10个样本
    asyncio.run(main_async(enable_quality_check, test_mode))
    print_time_cost()