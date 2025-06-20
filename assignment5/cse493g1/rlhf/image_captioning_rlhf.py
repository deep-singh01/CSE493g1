import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random

class ImageCaptioningRLHF:
    """
    RLHF trainer for image captioning with formality optimization.
    """
    
    def __init__(
        self,
        vlm_model_name: str = "Salesforce/blip-image-captioning-base",
        reward_model_name: str = "s-nlp/deberta-large-formality-ranker",
        device: Optional[str] = None
    ):
        """
        Initialize the RLHF trainer.
        
        Args:
            vlm_model_name: Name of the vision-language model
            reward_model_name: Name of the formality reward model
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load VLM
        print("Loading Vision-Language Model...")
        self.processor = BlipProcessor.from_pretrained(vlm_model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Load reward model
        print("Loading Formality Reward Model...")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name
        ).to(self.device)
        
        # Keep reference model for KL divergence
        self.reference_model = BlipForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.reference_model.eval()  # Keep reference model in eval mode
        
        print("✓ Models loaded successfully")
    
    def compute_formality_reward(self, captions: List[str]) -> torch.Tensor:
        """
        Compute formality rewards for generated captions.
        
        Args:
            captions: List of caption strings
            
        Returns:
            rewards: Tensor of shape (batch_size,) with formality scores
        """
        rewards = []
        for caption in captions:
            if not caption or len(caption.strip()) == 0:
                rewards.append(0.5)  # neutral score for empty captions
                continue
                
            inputs = self.reward_tokenizer(
                caption, return_tensors='pt', truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                scores = F.softmax(outputs.logits, dim=1)
                # Use index 0 for formality (model outputs are inverted)
                reward = scores[0][0].item()
                rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def generate_captions(
        self, 
        images: List[Image.Image], 
        prompts: List[str] = None,
        max_new_tokens: int = 25,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate captions for given images.
        
        Args:
            images: List of PIL Images
            prompts: Optional list of prompts (not used for BLIP)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            generated_ids: Tensor of generated token IDs
            captions: List of generated caption strings
        """
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate captions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        
        # Decode captions
        captions = [
            self.processor.decode(gen_id, skip_special_tokens=True) 
            for gen_id in generated_ids
        ]
        
        return generated_ids, captions

    def compute_reinforce_loss(
        self, 
        rewards: torch.Tensor, 
        generated_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss for image captioning.
        
        Args:
            rewards: Tensor of shape (batch_size,) with reward values
            generated_ids: Tensor of shape (batch_size, seq_len) with generated token IDs
            pixel_values: Tensor with processed image pixel values
            baseline: Optional baseline for variance reduction
            
        Returns:
            loss: REINFORCE loss scalar
            
        TODO: Implement REINFORCE loss computation for image captioning.
        
        Steps:
        1. Get model outputs by passing pixel_values and generated_ids to self.model
        2. Compute log probabilities from the logits
        3. Extract log probabilities for the generated tokens (handle sequence alignment)
        4. Apply baseline if provided: adjusted_rewards = rewards - baseline
        5. Compute REINFORCE loss: -adjusted_rewards * sum(log_probs_for_generated_tokens)
        6. Return the mean loss across the batch
        """
        ##############################################################################
        # TODO: Start of your code.                                                  #
        # Fill in the missing implementation by following the steps outlined above.  #
        # Hint: Compare with the reference implementation in the solution directory. #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss

    def compute_reinforce_loss_with_kl(
        self, 
        rewards: torch.Tensor,
        generated_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        kl_coeff: float = 0.1,
        baseline: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE loss with KL divergence regularization.
        
        Args:
            rewards: Tensor of shape (batch_size,) with reward values
            generated_ids: Tensor of shape (batch_size, seq_len) with generated token IDs
            pixel_values: Tensor with processed image pixel values
            kl_coeff: Coefficient for KL divergence term
            baseline: Optional baseline for variance reduction
            
        Returns:
            loss: Total loss (REINFORCE + KL)
            kl_divergence: KL divergence value
            
        TODO: Implement REINFORCE loss with KL divergence regularization.
        
        Steps:
        1. Compute REINFORCE loss using compute_reinforce_loss()
        2. Get log probabilities from current model
        3. Get log probabilities from reference model (use torch.no_grad())
        4. Compute KL divergence between current and reference model distributions
        5. Combine losses: total_loss = reinforce_loss + kl_coeff * kl_divergence
        6. Return total loss and KL divergence
        """
        ##############################################################################
        # TODO: Start of your code.                                                  #
        # Implement REINFORCE with KL regularization by following the steps above.    #
        # Hint: You can reuse compute_reinforce_loss() to get the base loss.          #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss, kl_divergence

    def sequence_log_prob(self, logits: torch.Tensor, input_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Helper function to compute sequence log probabilities."""
        log_probs = F.log_softmax(logits, dim=-1)
        tgt = input_ids[:, 1:]  # shift
        gather = torch.gather(log_probs[:, :-1], 2, tgt.unsqueeze(2)).squeeze(2)
        mask = (tgt != pad_id).float()
        seq_log_probs = (gather * mask).sum(dim=1)
        return seq_log_probs

    def compute_dpo_loss(
        self, 
        images: List[Image.Image], 
        chosen_captions: List[str], 
        rejected_captions: List[str], 
        beta: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DPO (Direct Preference Optimization) loss.
        
        Args:
            images: List of PIL Images
            chosen_captions: List of preferred captions
            rejected_captions: List of rejected captions
            beta: Temperature parameter for DPO
            
        Returns:
            loss: DPO loss scalar
            accuracy: Preference accuracy
            
        TODO: Implement DPO loss computation for image captioning.
        
        Steps:
        1. Process images and tokenize captions
        2. Get log probabilities from policy model for chosen and rejected captions
        3. Get log probabilities from reference model for chosen and rejected captions
        4. Compute log ratios: policy_logprobs - reference_logprobs
        5. Compute preference logits: beta * (chosen_ratio - rejected_ratio)
        6. Apply log-sigmoid and compute DPO loss: -log_sigmoid(preference_logits).mean()
        7. Compute accuracy: (preference_logits > 0).float().mean()
        8. Return loss and accuracy
        """
        ##############################################################################
        # TODO: Start of your code.                                                  #
        # Implement DPO loss computation by following the steps above.               #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss, accuracy

    def train_reinforce(
        self,
        images: List[Image.Image],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        use_kl: bool = False,
        kl_coeff: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train using REINFORCE algorithm.
        
        Args:
            images: List of training images
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            use_kl: Whether to use KL divergence regularization
            kl_coeff: KL divergence coefficient
            
        Returns:
            Dictionary with training metrics
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        losses = []
        rewards = []
        kl_divergences = [] if use_kl else None
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i:i+batch_size]
                
                # Generate captions
                generated_ids, captions = self.generate_captions(batch_images)
                
                # Compute rewards
                batch_rewards = self.compute_formality_reward(captions)
                
                # Process images for loss computation
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(self.device)
                
                # Compute loss
                if use_kl:
                    loss, kl_div = self.compute_reinforce_loss_with_kl(
                        batch_rewards, generated_ids, pixel_values, kl_coeff
                    )
                    kl_divergences.append(kl_div.item())
                else:
                    loss = self.compute_reinforce_loss(
                        batch_rewards, generated_ids, pixel_values
                    )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record metrics
                losses.append(loss.item())
                rewards.append(batch_rewards.mean().item())
        
        results = {
            'losses': losses,
            'rewards': rewards
        }
        
        if use_kl:
            results['kl_divergences'] = kl_divergences
            
        return results

    def train_dpo(
        self,
        preference_data: List[Dict],
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        beta: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Train using DPO algorithm.
        
        Args:
            preference_data: List of preference pairs with 'image', 'chosen', 'rejected'
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            beta: DPO temperature parameter
            
        Returns:
            Dictionary with training metrics
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        losses = []
        accuracies = []
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            for i in tqdm(range(0, len(preference_data), batch_size)):
                batch_data = preference_data[i:i+batch_size]
                
                batch_images = [item['image'] for item in batch_data]
                batch_chosen = [item['chosen'] for item in batch_data]
                batch_rejected = [item['rejected'] for item in batch_data]
                
                # Compute DPO loss
                loss, accuracy = self.compute_dpo_loss(
                    batch_images, batch_chosen, batch_rejected, beta
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record metrics
                losses.append(loss.item())
                accuracies.append(accuracy.item())
        
        return {
            'losses': losses,
            'accuracies': accuracies
        }

    def evaluate(self, images: List[Image.Image]) -> float:
        """
        Evaluate the model on a set of images.
        
        Args:
            images: List of evaluation images
            
        Returns:
            Average formality reward
        """
        self.model.eval()
        
        total_reward = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(images), 8):  # Use smaller batch for evaluation
                batch_images = images[i:i+8]
                
                # Generate captions
                _, captions = self.generate_captions(batch_images)
                
                # Compute rewards
                rewards = self.compute_formality_reward(captions)
                
                total_reward += rewards.mean().item()
                num_batches += 1
        
        return total_reward / num_batches if num_batches > 0 else 0.0 