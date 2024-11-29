import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
from .multihead_flashdiff_1 import MultiheadFlashDiff1


class StockSelectionHead(nn.Module):
    """Assigns logits to each stock based on relevance to the input text."""
    def __init__(self, d_model: int, hidden_dim: int, num_stocks: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_stocks)
        
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

class CustomDecoderLayer(nn.Module):
    """Modified transformer decoder layer with cross-attention before self-attention."""
    def __init__(self, hidden_dim: int, num_heads: int, depth: int, attn_dropout: float = 0.1, ff_dropout: float = 0.1):
        super().__init__()
        
        # Cross attention (to attend to encoded text)
        self.cross_attention = MultiheadFlashDiff1(
            embed_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Non-causal self attention (stocks attending to each other)
        self.self_attention = MultiheadFlashDiff1(
            embed_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads
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
    """Full decoder with multiple layers."""
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(hidden_dim, num_heads, depth=i, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
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
            self_attn_mask: Optional mask for self attention
            cross_attn_mask: Optional mask for cross attention
            
        Returns:
            output: Final stock embeddings
        """
        for layer in self.layers:
            x = layer(
                x, encoder_output
            )
        return x

class MultiStockPredictor(nn.Module):
    """Complete model combining encoder, stock selection, and decoder."""
    def __init__(
        self,
        num_stocks: int,
        num_decoder_layers: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        encoder_name: str = "FinText/FinText-Base-2007"
    ):
        super().__init__()
        
        # Load pretrained encoder and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Stock selection head
        self.stock_selector = StockSelectionHead(hidden_dim, num_stocks)
        
        # Stock embedding layer (convert stock indices to embeddings)
        self.stock_embeddings = nn.Embedding(num_stocks, hidden_dim)
        
        # Decoder
        self.decoder = StockDecoder(
            num_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        
                # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden_dim * 4, 1)
        )  # Predicts return for each stock
        
    def forward(
        self,
        input: dict[str, torch.Tensor],  # Output from tokenizer
        k: int  # Number of stocks to select
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Input token ids (batch_size, seq_length)
            attention_mask: Attention mask for input (batch_size, seq_length)
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