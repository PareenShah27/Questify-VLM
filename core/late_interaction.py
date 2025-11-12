"""
core/late_interaction.py - Late Interaction Mechanism

Implements ColBERT-style late interaction for efficient similarity computation
between query tokens and document patches.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class LateInteractionScorer:
    """
    Late Interaction scoring mechanism for multimodal retrieval.
    
    Implements the MaxSim operation from ColBERT:
    For each query token, compute maximum similarity with all document tokens,
    then sum across query tokens to get final score.
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        """
        Initialize Late Interaction Scorer.
        
        Args:
            similarity_metric: 'cosine' or 'dot_product'
        """
        self.similarity_metric = similarity_metric
    
    def compute_late_interaction_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute late interaction score between query and document.
        
        Args:
            query_embeddings: (num_queries, query_len, dim)
            doc_embeddings: (num_docs, doc_len, dim)
            query_mask: Optional mask for query padding
            doc_mask: Optional mask for document padding
        
        Returns:
            Similarity scores (num_queries, num_docs)
        """
        # Normalize embeddings if using cosine similarity
        if self.similarity_metric == "cosine":
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        
        # Compute all pairwise similarities
        # (num_queries, query_len, dim) @ (num_docs, dim, doc_len)
        # = (num_queries, num_docs, query_len, doc_len)
        scores = torch.einsum(
            'qld,dkD->qldk',
            query_embeddings,
            doc_embeddings.transpose(-2, -1)
        )
        
        # Apply document mask if provided
        if doc_mask is not None:
            scores = scores.masked_fill(~doc_mask.unsqueeze(0).unsqueeze(2), float('-inf'))
        
        # MaxSim: For each query token, find max similarity across doc tokens
        max_scores = scores.max(dim=-1)[0]  # (num_queries, num_docs, query_len)
        
        # Apply query mask if provided
        if query_mask is not None:
            max_scores = max_scores.masked_fill(~query_mask.unsqueeze(1), 0.0)
        
        # Sum across query tokens
        final_scores = max_scores.sum(dim=-1)  # (num_queries, num_docs)
        
        return final_scores
    
    def compute_maxsim(
        self,
        query_embedding: torch.Tensor,
        doc_embedding: torch.Tensor
    ) -> float:
        """
        Compute MaxSim score for a single query-document pair.
        
        Args:
            query_embedding: (query_len, dim)
            doc_embedding: (doc_len, dim)
        
        Returns:
            Similarity score (scalar)
        """
        # Normalize if using cosine
        if self.similarity_metric == "cosine":
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            doc_embedding = F.normalize(doc_embedding, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(
            query_embedding,
            doc_embedding.transpose(0, 1)
        )  # (query_len, doc_len)
        
        # MaxSim operation
        max_sim_per_query_token = sim_matrix.max(dim=1)[0]  # (query_len,)
        score = max_sim_per_query_token.sum().item()
        
        return score
    
    def batch_compute_scores(
        self,
        query_embeddings: List[torch.Tensor],
        doc_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute scores for multiple query-document pairs efficiently.
        
        Args:
            query_embeddings: List of query embeddings
            doc_embeddings: List of document embeddings
        
        Returns:
            Score matrix (num_queries, num_docs)
        """
        num_queries = len(query_embeddings)
        num_docs = len(doc_embeddings)
        
        scores = torch.zeros(num_queries, num_docs)
        
        for i, q_emb in enumerate(query_embeddings):
            for j, d_emb in enumerate(doc_embeddings):
                scores[i, j] = self.compute_maxsim(q_emb, d_emb)
        
        return scores


class TokenPooler:
    """
    Token pooling mechanism to reduce embedding size while maintaining performance.
    Based on Hierarchical Token Pooling from ColPali paper.
    """
    
    def __init__(self, pool_factor: int = 2):
        """
        Initialize Token Pooler.
        
        Args:
            pool_factor: Factor by which to reduce sequence length
        """
        self.pool_factor = pool_factor
    
    def pool_embeddings(
        self,
        embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool embeddings to reduce sequence length.
        
        Args:
            embeddings: (batch_size, seq_len, dim)
            padding_mask: Optional mask for padding tokens
        
        Returns:
            Pooled embeddings with reduced sequence length
        """
        batch_size, seq_len, dim = embeddings.shape
        
        # Calculate new sequence length
        new_seq_len = seq_len // self.pool_factor
        
        if padding_mask is not None:
            # Mask padding before pooling
            embeddings = embeddings.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        
        # Reshape for pooling
        # (batch_size, new_seq_len, pool_factor, dim)
        pooled_shape = (batch_size, new_seq_len, self.pool_factor, dim)
        
        # Truncate if needed
        truncate_len = new_seq_len * self.pool_factor
        embeddings_truncated = embeddings[:, :truncate_len, :]
        
        # Reshape and pool (mean)
        reshaped = embeddings_truncated.view(pooled_shape)
        pooled = reshaped.mean(dim=2)  # Average over pool_factor dimension
        
        return pooled
    
    def hierarchical_pool(
        self,
        embeddings: List[torch.Tensor],
        padding_side: str = "right"
    ) -> List[torch.Tensor]:
        """
        Apply hierarchical pooling to a list of variable-length embeddings.
        
        Args:
            embeddings: List of embedding tensors
            padding_side: 'left' or 'right'
        
        Returns:
            List of pooled embeddings
        """
        pooled_embeddings = []
        
        for emb in embeddings:
            # Determine actual length (exclude padding)
            actual_len = emb.shape[0]
            
            # Pool the embedding
            if actual_len >= self.pool_factor:
                # Can pool
                new_len = actual_len // self.pool_factor
                truncate_len = new_len * self.pool_factor
                
                emb_truncated = emb[:truncate_len]
                reshaped = emb_truncated.view(new_len, self.pool_factor, -1)
                pooled = reshaped.mean(dim=1)
                
                pooled_embeddings.append(pooled)
            else:
                # Too short to pool, keep as is
                pooled_embeddings.append(emb)
        
        return pooled_embeddings


class SimilarityMap:
    """
    Generate interpretability maps showing which document regions match query tokens.
    """
    
    @staticmethod
    def generate_similarity_map(
        query_embedding: torch.Tensor,
        doc_embedding: torch.Tensor,
        doc_patches_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate similarity map for visualization.
        
        Args:
            query_embedding: (query_len, dim)
            doc_embedding: (doc_len, dim)
            doc_patches_shape: (height, width) of document patch grid
        
        Returns:
            Similarity map (query_len, height, width)
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(
            F.normalize(query_embedding, p=2, dim=-1),
            F.normalize(doc_embedding, p=2, dim=-1).transpose(0, 1)
        )  # (query_len, doc_len)
        
        # Reshape to spatial dimensions
        h, w = doc_patches_shape
        assert sim_matrix.shape[1] == h * w, "Document length must match patch grid"
        
        similarity_map = sim_matrix.view(-1, h, w)
        
        return similarity_map
    
    @staticmethod
    def get_top_k_patches(
        similarity_map: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Get coordinates of top-k most similar patches for each query token.
        
        Args:
            similarity_map: (query_len, height, width)
            k: Number of top patches to retrieve
        
        Returns:
            List of top-k patch coordinates per query token
        """
        query_len, h, w = similarity_map.shape
        
        top_patches = []
        for q_idx in range(query_len):
            flat_sim = similarity_map[q_idx].flatten()
            top_k_indices = torch.topk(flat_sim, min(k, len(flat_sim))).indices
            
            # Convert to 2D coordinates
            coords = [(idx.item() // w, idx.item() % w) for idx in top_k_indices]
            top_patches.append(coords)
        
        return top_patches