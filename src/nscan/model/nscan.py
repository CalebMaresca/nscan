import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Tuple, Optional
from .layers import StockDecoder, create_predictor, StockSelectionHead
from ..utils import confidence_weighted_loss

@dataclass
class StockPredictionOutput(ModelOutput):
    """
    Output type of NSCAN.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Confidence weighted MSE loss if `returns` provided.
        predictions (`torch.FloatTensor` of shape `(batch_size, num_stocks)`):
            Predicted returns for each stock.
        confidences (`torch.FloatTensor` of shape `(batch_size, num_stocks)`):
            Confidence scores for each prediction.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of intermediate hidden states.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of attention weights.
    """
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None
    confidences: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class NSCANConfig(PretrainedConfig):
    model_type = "nscan"
    
    def __init__(
        self,
        num_stocks: int = 500,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        num_pred_layers: int = 3,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_flash: bool = True,
        encoder_name: str = "FinText/FinText-Base-2007",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_stocks = num_stocks
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.num_pred_layers = num_pred_layers
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.use_flash = use_flash
        self.encoder_name = encoder_name

class NSCANPreTrainedModel(PreTrainedModel):
    """
    Base class for NSCAN model.
    """
    config_class = NSCANConfig
    base_model_prefix = "nscan"


class NSCAN(NSCANPreTrainedModel):
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
    def __init__(self, config: NSCANConfig):
        super().__init__(config)
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Stock embedding layer (convert stock indices to embeddings)
        self.stock_embeddings = nn.Embedding(
            num_embeddings=config.num_stocks + 1,
            embedding_dim=hidden_dim,
            padding_idx=0
)
        
        # Decoder
        self.decoder = StockDecoder(
            num_layers=config.num_decoder_layers,
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            use_flash=config.use_flash
        )
        
        # Return prediction head
        self.return_predictor = create_predictor(
            input_dim=hidden_dim,
            num_layers=config.num_pred_layers,
            ff_dropout=config.ff_dropout
        )
        
        # Confidence prediction head
        self.confidence_predictor = create_predictor(
            input_dim=hidden_dim,
            num_layers=config.num_pred_layers,
            ff_dropout=config.ff_dropout
        )
        
    def forward(
        self,
        input: dict[str, torch.Tensor],  # Output from tokenizer
        stock_indices: torch.Tensor,  # Shape: (batch_size, num_stocks_per_date)
        returns: Optional[torch.Tensor] = None
    ) -> StockPredictionOutput:
        """Forward pass of the model.
        
        Args:
            input: Output from tokenizer containing input_ids, attention_mask, etc.
            stock_indices: Stock indices tensor of shape (batch_size, num_stocks_per_date)
            returns: Optional ground truth returns for training
            
        Returns:
            StockPredictionOutput containing predictions, confidences, and optionally loss
        """
        predictions, confidences = self._forward(input, stock_indices)
        
        output = StockPredictionOutput(
            predictions=predictions,
            confidences=confidences
        )
        
        if returns is not None:
            output.loss = confidence_weighted_loss(predictions, returns, confidences)
            
        return output
    
    def _forward(self, input, stock_indices):
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