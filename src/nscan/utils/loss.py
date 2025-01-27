import torch

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