import torch
import torch.nn as nn
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