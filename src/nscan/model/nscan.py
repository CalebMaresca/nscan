import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple
from .layers import StockDecoder, create_predictor, StockSelectionHead

class NSCAN(nn.Module):
    """News-Stock Cross-Attention Network for multi-stock prediction.
    
    This model combines a pretrained language model with a custom decoder 
    architecture to predict stock returns based on financial news articles.
    
    Attributes:
        encoder: Pretrained language model for processing news text
        stock_embeddings: Learnable embeddings for each stock
        decoder: Custom transformer decoder for cross-attention
        return_predictor: MLP for predicting returns
        confidence_predictor: MLP for predicting confidence scores
        
    Args:
        num_stocks (int): Total number of unique stocks in dataset
        num_decoder_layers (int): Number of transformer decoder layers
        num_heads (int): Number of attention heads in decoder
        num_pred_layers (int): Number of layers in prediction MLPs
        attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.1
        ff_dropout (float, optional): Dropout rate for feedforward layers. Defaults to 0.1
        use_flash (bool, optional): Whether to use flash attention. Defaults to True
        encoder_name (str, optional): HuggingFace model name. Defaults to "FinText/FinText-Base-2007"
    """
    def __init__(
        self,
        num_stocks: int,
        num_decoder_layers: int,
        num_heads: int,
        num_pred_layers: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_flash: bool = True,
        encoder_name: str = "FinText/FinText-Base-2007"
    ):
        super().__init__()
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Stock embedding layer (convert stock indices to embeddings)
        self.stock_embeddings = nn.Embedding(
            num_embeddings=num_stocks + 1,
            embedding_dim=hidden_dim,
            padding_idx=0
)
        

def confidence_weighted_loss(predictions: torch.Tensor, 
                           targets: torch.Tensor, 
                           confidences: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence-weighted MSE loss.

    This loss function weights the squared error for each prediction by the model's
    confidence in that prediction. This allows the model to learn to assign higher
    confidence to predictions it can make more accurately.
    
    Args:
        predictions: Predicted returns (batch_size, num_stocks)
        targets: Actual returns (batch_size, num_stocks)
        confidences: Confidence scores (batch_size, num_stocks)
    
    Returns:
        loss: Weighted MSE loss

    Note:
    - Expects confidences to sum to 1.0 across stocks for each batch
    - Includes safety checks for NaN values and extreme predictions
    - Clips predictions and targets to [-100, 100] range
    """

    # Check for NaN inputs
    if torch.isnan(predictions).any():
        print("NaN detected in predictions")
    if torch.isnan(targets).any():
        print("NaN detected in targets")
    if torch.isnan(confidences).any():
        print("NaN detected in confidences")

    # Clip predictions and targets to prevent extreme values
    predictions = torch.clamp(predictions, -100, 100)
    targets = torch.clamp(targets, -100, 100)

    squared_errors = (predictions - targets) ** 2
    weighted_errors = squared_errors * confidences

    # Sanity check - confidences should sum to 1.0 per batch
    batch_sums = confidences.sum(dim=-1)
    if not torch.allclose(batch_sums, torch.ones_like(batch_sums)):
        print(f"Warning: Confidence sums not 1.0: {batch_sums}")

    return weighted_errors.sum(dim=-1).mean()


class NSCAN_old(nn.Module):
    """Older version of NSCAN."""
    def __init__(
        self,
        num_stocks: int,
        num_decoder_layers: int,
        num_heads: int,
        num_pred_layers: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_flash: bool = True,
        encoder_name: str = "FinText/FinText-Base-2007"
    ):
        super().__init__()
        
        # Load pretrained encoder
        #self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Stock selection head
        self.stock_selector = StockSelectionHead(hidden_dim, num_stocks)
        
        # Stock embedding layer (convert stock indices to embeddings)
        self.stock_embeddings = nn.Embedding(
            num_embeddings=num_stocks + 1,
            embedding_dim=hidden_dim,
            padding_idx=0
)
        
        # Decoder
        self.decoder = StockDecoder(
            num_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_flash=use_flash
        )
        
        # Return prediction head
        self.predictor = create_predictor(
            input_dim=hidden_dim,
            num_layers=num_pred_layers,
            ff_dropout=ff_dropout
        )
        
    def forward(
        self,
        input: dict[str, torch.Tensor],  # Output from tokenizer
        k: int  # Number of stocks to select
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Input token ids (batch_size, seq_length)
            k: Number of stocks to select and predict
            
        Returns:
            predictions: Predicted returns for selected stocks
            stock_logits: Logits for stock selection
        """
        # Encode text
        encoder_output = self.encoder(**input).last_hidden_state
        
        # Get CLS token for stock selection
        cls_token = encoder_output[:, 0]
        
        # Select stocks
        stock_logits = self.stock_selector(cls_token)
        top_k_values, top_k_indices = torch.topk(stock_logits, k)
        
        # Get stock embeddings
        stock_embeddings = self.stock_embeddings(top_k_indices)  # (batch_size, k, hidden_dim)
        
        # Pass through decoder
        decoded_stocks = self.decoder(
            stock_embeddings,
            encoder_output
        )
        
        # Predict returns
        predictions = self.predictor(decoded_stocks).squeeze(-1)  # (batch_size, k)
        
        return predictions, stock_logits