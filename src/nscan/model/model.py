import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple
from .multihead_diff_2 import MultiheadDiff2


class CustomDecoderLayer(nn.Module):
    """Modified transformer decoder layer with cross-attention before self-attention.
    
    This layer differs from standard transformer decoders by applying cross-attention
    to the encoded text before self-attention between stocks. This architecture choice
    helps stocks first gather relevant information from the text before interacting
    with each other.

    Additionally, it uses differential attention for improved performance.
    
    Args:
        hidden_dim (int): Dimension of the input embeddings
        num_heads (int): Number of attention heads
        depth (int): Position of this layer in the decoder stack (used for initialization)
        attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.1
        ff_dropout (float, optional): Dropout rate for feed-forward layers. Defaults to 0.1
        use_flash (bool, optional): Whether to use flash attention. Defaults to True
    """
    def __init__(self, hidden_dim: int, num_heads: int, depth: int, attn_dropout: float = 0.1, ff_dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        
        # Cross attention (to attend to encoded text)
        self.cross_attention = MultiheadDiff2(
            embed_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            use_flash=use_flash
        )
        
        # Non-causal self attention (stocks attending to each other)
        self.self_attention = MultiheadDiff2(
            embed_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            use_flash=use_flash
        )
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Stock embeddings (batch_size, num_selected_stocks, hidden_dim)
            encoder_output: Encoded text (batch_size, seq_length, hidden_dim)
            
        Returns:
            output: Processed stock embeddings
        """
        # Cross attention first
        residual = x
        x = self.norm1(x)
        cross_attn_output = self.cross_attention(
            x=x,
            encoder_out=encoder_output
        )
        x = residual + self.attn_dropout(cross_attn_output)
        
        # Then self attention
        residual = x
        x = self.norm2(x)
        self_attn_output = self.self_attention(
            x=x
        )
        x = residual + self.attn_dropout(self_attn_output)
        
        # Finally feed forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff_dropout(self.feed_forward(x))
        
        return x

class StockDecoder(nn.Module):
    """Full decoder stack for processing stock embeddings with text context.
    
    Consists of multiple CustomDecoderLayers that iteratively refine stock
    representations using both text information (cross-attention) and
    inter-stock relationships (self-attention).
    
    Args:
        num_layers (int): Number of decoder layers to stack
        hidden_dim (int): Dimension of the input embeddings
        num_heads (int): Number of attention heads per layer
        attn_dropout (float, optional): Dropout rate for attention. Defaults to 0.1
        ff_dropout (float, optional): Dropout rate for feed-forward layers. Defaults to 0.1
        use_flash (bool, optional): Whether to use flash attention. Defaults to True
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(hidden_dim, num_heads, depth=i, attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_flash=use_flash)
            for i in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Stock embeddings (batch_size, num_selected_stocks, hidden_dim)
            encoder_output: Encoded text (batch_size, seq_length, hidden_dim)
            
        Returns:
            output: Final stock embeddings
        """
        for layer in self.layers:
            x = layer(
                x, encoder_output
            )
        return x

def create_predictor(input_dim: int, num_layers: int, ff_dropout: float) -> nn.Sequential:
    """Creates a multi-layer perceptron for prediction tasks.
    
    Constructs a deep neural network with residual connections and dropout
    for predicting either returns or confidence scores.
    
    Args:
        input_dim (int): Dimension of input features
        num_layers (int): Number of hidden layers
        ff_dropout (float): Dropout rate for regularization
        
    Returns:
        nn.Sequential: The constructed MLP
    """
    layers = []
    # First layer
    layers.extend([
        nn.Linear(input_dim, 4 * input_dim),
        nn.GELU(),
        nn.Dropout(ff_dropout)
    ])
    # Middle layers
    for _ in range(num_layers - 2):
        layers.extend([
            nn.Linear(4 * input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout)
        ])
    # Final layer
    layers.append(nn.Linear(4 * input_dim, 1))
    return nn.Sequential(*layers)


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
        self.return_predictor = create_predictor(
            input_dim=hidden_dim,
            num_layers=num_pred_layers,
            ff_dropout=ff_dropout
        )
        
        # Confidence prediction head
        self.confidence_predictor = create_predictor(
            input_dim=hidden_dim,
            num_layers=num_pred_layers,
            ff_dropout=ff_dropout
        )
        
    def forward(
        self,
        input: dict[str, torch.Tensor],  # Output from tokenizer
        stock_indices: torch.Tensor,  # Shape: (batch_size, num_stocks_per_date)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Input token ids (batch_size, seq_length)
            
        Returns:
            predictions: Predicted returns for all stocks (batch_size, num_stocks)
            confidences: Confidence scores for all stocks (batch_size, num_stocks)
        """
        # Create padding mask (1 for real stocks, 0 for padding)
        padding_mask = (stock_indices != 0).float()  # Or whatever padding index you used

        # Get stock embeddings for the valid stocks
        stock_embeddings = self.stock_embeddings(stock_indices)  # (batch_size, num_stocks_per_date, hidden_dim)

        # Encode text
        encoder_output = self.encoder(**input).last_hidden_state
        
        # Pass through decoder
        decoded_stocks = self.decoder(
            stock_embeddings,
            encoder_output
        )
        
        # Predict returns and confidences
        predictions = self.return_predictor(decoded_stocks).squeeze(-1) * padding_mask  # (batch_size, num_stocks)
        confidence_logits = self.confidence_predictor(decoded_stocks).squeeze(-1) * padding_mask  # (batch_size, num_stocks)
        
        # Normalize confidence scores using softmax
        confidences = torch.softmax(confidence_logits, dim=-1)
        
        return predictions, confidences

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
    
    class StockSelectionHead(nn.Module):
        """Assigns logits to each stock based on relevance to the input text.
        
        A simple feed-forward network that takes the CLS token embedding and outputs
        logits for each stock, representing their relevance to the input text.

        Used in the older NSCAN model.
        
        Args:
            hidden_dim (int): Dimension of the input embeddings
            num_stocks (int): Total number of stocks to score
    """
        def __init__(self, hidden_dim: int, num_stocks: int):
            super().__init__()
            self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(4 * hidden_dim, num_stocks)
            
        def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
            """
            Args:
                cls_token: Tensor of shape (batch_size, hidden_dim)
            Returns:
                logits: Tensor of shape (batch_size, num_stocks)
            """
            x = self.linear1(cls_token)
            x = self.relu(x)
            logits = self.linear2(x)
            return logits