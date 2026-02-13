"""
Cosmos-Reason2-8B Model Wrapper for EmbodiedBench.

Based on the official setup from:
https://huggingface.co/nvidia/Cosmos-Reason2-8B

The model is a Qwen3-VL-8B-Instruct derivative, using the same architecture.
Requires ~32GB GPU (split across 2x 16GB GPUs).
"""

import torch
import transformers
import os
import json
import base64
from PIL import Image
import io
import re


import faulthandler
faulthandler.enable()

class CosmosReason2Model:
    """Wrapper for loading and running inference with Cosmos-Reason2-8B."""
    
    def __init__(self, model_name="nvidia/Cosmos-Reason2-8B", device_map="auto", dtype=torch.bfloat16):
        print(f"Loading model: {model_name}...", flush=True)
        print(f"  device_map={device_map}, dtype={dtype}", flush=True)
        
        self.model_name = model_name
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        
        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        print(f"Model loaded successfully on {self.model.device if hasattr(self.model, 'device') else 'multiple devices'}", flush=True)

    def respond(self, system_prompt, user_text, image_paths=None, max_new_tokens=4096):
        """
        Generate a response given system prompt, user text, and optional images.
        
        Args:
            system_prompt: System-level instruction
            user_text: User query/instruction  
            image_paths: List of image file paths to include, or None
            max_new_tokens: Max tokens for generation
            
        Returns:
            tuple: (full_output, thinking, answer, action_content) - parsed response
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
        ]
        
        # Build user content
        user_content = []
        
        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    # Load as PIL Image to avoid processor path/URI issues
                    img = Image.open(img_path).convert("RGB")
                    user_content.append({
                        "type": "image",
                        "image": img,
                    })
        
        # Add text
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for deterministic action selection
                temperature=None,
                top_p=None,
            )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Parse thinking, answer and action
        thinking, answer, action_content = self._parse_response(output_text)
        
        return output_text, thinking, answer, action_content

    def _parse_response(self, text):
        """Parse response for both tagged format and numbered sequence format."""
        thinking = ""
        answer = ""
        action_content = ""
        
        # Method 1: Try numbered sequence (1. [Reasoning:] ... 2. [Answer:] ... 3. [Action IDs:] ...)
        # We look for "1.", "2.", "3." markers, optionally followed by labels
        seq_match = re.search(r'1\.\s*(?:Reasoning:)?\s*(.*?)\s*2\.\s*(?:Answer:)?\s*(.*?)\s*3\.\s*(?:Action(?: IDs)?:)?\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if seq_match:
            thinking = seq_match.group(1).strip()
            answer = seq_match.group(2).strip()
            action_content = seq_match.group(3).strip()
            return thinking, answer, action_content

        # Method 2: Fallback to tag parsing (in case model ignores instructions or reverts)
        # Extract thinking
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # Extract answer
        ans_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if ans_match:
            answer = ans_match.group(1).strip()
        
        # Extract action
        act_match = re.search(r'<action>(.*?)(?:</action>|$)', text, re.DOTALL)
        if act_match:
            action_content = act_match.group(1).strip()
            
        # Global Fallbacks
        if not answer and not action_content:
            if think_match:
                # Part after think
                rem = text[think_match.end():].strip()
                # Check for tags in remainder
                if "<answer>" in rem and not answer:
                    ans_m = re.search(r'<answer>(.*?)(?:</answer>|$)', rem, re.DOTALL)
                    if ans_m: answer = ans_m.group(1).strip()
                if "<action>" in rem and not action_content:
                    act_m = re.search(r'<action>(.*?)(?:</action>|$)', rem, re.DOTALL)
                    if act_m: action_content = act_m.group(1).strip()
                
                if not answer:
                    answer = rem
            elif not seq_match:
                # If no structure found, treat whole text as answer (likely JSON)
                answer = text.strip()
                
        return thinking, answer, action_content
