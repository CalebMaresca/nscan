from nscan.model.model import MultiStockPredictor

def analyze_model(model):
    """Analyze model parameters and memory usage by component"""
    
    def get_param_size(params):
        return sum(p.numel() * p.element_size() for p in params) / (1024 * 1024)  # Size in MB
    
    def get_param_count(params):
        return sum(p.numel() for p in params)
    
    # Analyze each component
    components = {
        'encoder': model.encoder,
        'stock_selector': model.stock_selector,
        'stock_embeddings': model.stock_embeddings,
        'decoder': model.decoder,
        'predictor': model.predictor
    }
    
    total_params = 0
    total_size = 0
    
    print(f"{'Component':<20} {'Parameters':<15} {'Size (MB)':<10}")
    print("-" * 45)
    
    for name, component in components.items():
        params = list(component.parameters())
        param_count = get_param_count(params)
        size_mb = get_param_size(params)
        
        total_params += param_count
        total_size += size_mb
        
        print(f"{name:<20} {param_count:<15,d} {size_mb:.2f}")
    
    print("-" * 45)
    print(f"{'Total':<20} {total_params:<15,d} {total_size:.2f}")

# Usage example:
model = MultiStockPredictor(num_stocks=500, num_decoder_layers=3, num_heads=4)
analyze_model(model)
