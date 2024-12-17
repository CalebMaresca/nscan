import os
import json
import torch
from torch.utils.data import DataLoader
import backtrader as bt
import pandas as pd
from collections import defaultdict
from datasets import load_from_disk
from nscan.utils.data import collate_fn, NewsReturnDataset
from nscan.model import MultiStockPredictorWithConfidence, confidence_weighted_loss
from torch.amp import autocast

def move_batch(batch, device):
        # Move to device
        return {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'stock_indices': batch['stock_indices'].to(device),
            'returns': batch['returns'].to(device)
        }

def evaluate_model(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_predictions = []
    all_confidences = []
    all_returns = []
    all_dates = []
    all_stock_indices = []
    total_loss = 0
    
    with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                    
                device_batch = move_batch(batch, device)
                
                with autocast(device_type=str(device)):
                    predictions, confidences = model(
                        input={
                            'input_ids': device_batch['input_ids'],
                            'attention_mask': device_batch['attention_mask']
                        },
                        stock_indices=device_batch['stock_indices']
                    )
                    
                    loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
                    total_loss += loss.item()
    # with torch.no_grad():
    #     for batch in test_loader:
    #         device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) 
    #                       else v for k, v in batch.items()}
            
    #         predictions, confidences = model(
    #             input={
    #                 'input_ids': device_batch['input_ids'],
    #                 'attention_mask': device_batch['attention_mask']
    #             },
    #             stock_indices=device_batch['stock_indices']
    #         )
            
    #         loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
    #         total_loss += loss.item()
            
            # Store predictions and metadata
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())
            all_returns.append(device_batch['returns'].cpu())
            all_dates.extend(device_batch['date'])
            all_stock_indices.append(device_batch['stock_indices'].cpu())
    
    return {
        'test_loss': total_loss / len(test_loader),
        'predictions': torch.cat(all_predictions),
        'confidences': torch.cat(all_confidences),
        'returns': torch.cat(all_returns),
        'dates': all_dates,
        'stock_indices': torch.cat(all_stock_indices)
    }

class NewsBasedStrategy(bt.Strategy):
    params = (
        ('initial_cash', 100000),
    )
    
    def __init__(self):
        self.predictions_by_date = self.datas[0].predictions_by_date
        self.confidences_by_date = self.datas[0].confidences_by_date
        self.stock_indices = self.datas[0].stock_indices
        
    def next(self):
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        
        if current_date in self.predictions_by_date:
            predictions = self.predictions_by_date[current_date]
            confidences = self.confidences_by_date[current_date]
            
            # Combine predictions for the same day using confidence-weighted average
            weighted_predictions = defaultdict(list)
            weighted_confidences = defaultdict(list)
            
            for pred, conf, stocks in zip(predictions, confidences, self.stock_indices):
                for p, c, s in zip(pred, conf, stocks):
                    if s != 0:  # Skip padding
                        weighted_predictions[s].append(p * c)
                        weighted_confidences[s].append(c)
            
            # Calculate final predictions
            final_predictions = {}
            for stock, preds in weighted_predictions.items():
                confs = weighted_confidences[stock]
                final_predictions[stock] = sum(preds) / sum(confs)
            
            # Sort stocks by predicted returns
            sorted_stocks = sorted(
                final_predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Invest in top N stocks with highest predicted returns
            N = 10  # Number of stocks to invest in
            total_value = self.broker.getvalue()
            position_size = total_value / N
            
            # Close existing positions
            for data in self.datas:
                if self.getposition(data).size != 0:
                    self.close(data)
            
            # Open new positions
            for stock, pred in sorted_stocks[:N]:
                if pred > 0:  # Only go long if predicted return is positive
                    try:
                        stock_data = self.getdatabyname(str(stock))
                        if stock_data is None:
                            continue
                        if not math.isnan(stock_data.close[0]):
                            self.buy(data=stock_data, 
                                size=position_size/stock_data.close[0])
                    except Exception as e:
                        print(f"Error trading stock {stock}: {e}")

class ReturnsData(bt.feeds.PandasData):
    """Custom data feed that works with returns instead of prices"""
    
    params = (
        ('returns', 'Returns'),  # Name of returns column in DataFrame
        ('openinterest', None),  # Not using these fields
        ('volume', None),
        ('high', None),
        ('low', None),
        ('open', None),
        ('close', None),
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start with price of 1, will multiply by (1 + return) each day
        self._price = 1.0
        
    def _load(self):
        # Override _load to calculate synthetic prices from returns
        ret = super()._load()
        if ret:
            if len(self) > 1:  # After first day
                self._price *= (1.0 + self.lines.returns[0])
            # Set the close price
            self.lines.close[0] = self._price
        return ret

def run_backtest(test_results, returns_by_year, sp500_by_year):
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(NewsBasedStrategy)
    
    # Prepare data
    predictions_by_date = defaultdict(list)
    confidences_by_date = defaultdict(list)
    stock_indices_by_date = defaultdict(list)
    
    for date, pred, conf, stocks in zip(
        test_results['dates'],
        test_results['predictions'],
        test_results['confidences'],
        test_results['stock_indices']
    ):
        predictions_by_date[date].append(pred)
        confidences_by_date[date].append(conf)
        stock_indices_by_date[date].append(stocks)
    
    # Add data feeds
    for year in returns_by_year:
        for stock in sp500_by_year[year]:
            # Convert returns to DataFrame if it's a Series
            stock_data = returns_by_year[year][stock]
            if not isinstance(stock_data, pd.DataFrame):
                stock_data = stock_data.to_frame(name='Returns')
            
            data = ReturnsData(
                dataname=stock_data,
                name=str(stock),
                returns='Returns'  # Specify column name for returns
            )
            data.predictions_by_date = predictions_by_date
            data.confidences_by_date = confidences_by_date
            data.stock_indices = stock_indices_by_date
            cerebro.adddata(data)
    
    # Set starting cash
    cerebro.broker.setcash(100000)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    
    # Run backtest
    results = cerebro.run()
    
    # Print results
    strat = results[0]
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print(f'Sharpe Ratio: {strat.analyzers.sharperatio.get_analysis()["sharperatio"]:.2f}')
    print(f'Return: {strat.analyzers.returns.get_analysis()["rtot"]:.2%}')
    print(f'Max Drawdown: {strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2%}')
    
    return results

def load_returns_and_sp500_data(years, data_dir):
    # Load returns data by year
    returns_by_year = {}
    sp500_by_year = {}
    
    for year in years:
        # Load returns DataFrame for this year
        file_path = os.path.join(data_dir, f"{year}_returns.csv")
        df = pd.read_csv(file_path)

        # Check raw data before pivot
        print(f"\nYear {year}:")
        print(f"Raw data NaN count: {df['DlyRet'].isna().sum()}")
        
        # Pivot the data to get dates as rows and PERMNOs as columns
        returns_df = df.pivot(
            index='DlyCalDt', 
            columns='PERMNO', 
            values='DlyRet'
        ).sort_index(axis=1)  # Sort columns (PERMNOs)

        # Check pivoted data
        print(f"Pivoted data NaN count: {returns_df.isna().sum().sum()}")
        print(f"Total cells: {returns_df.size}")
        print(f"NaN percentage: {(returns_df.isna().sum().sum() / returns_df.size) * 100:.2f}%")

        # Fill NaN values with 0 (or another appropriate value)
        returns_df = returns_df.fillna(0)
        
        returns_by_year[str(year)] = returns_df
        # Get unique sorted PERMNOs for this year
        sp500_by_year[str(year)] = sorted(df['PERMNO'].unique().tolist())
        
    return returns_by_year, sp500_by_year

def test_model(test_years, checkpoint_path, config_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the preprocessed datasets and metadata
    dataset_dict = load_from_disk(os.path.join(data_dir, "preprocessed_datasets"))  # This loads the DatasetDict
    test_dataset = NewsReturnDataset(dataset_dict['test'])  # Get the test split

    # Load returns data
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(test_years, os.path.join(data_dir, "returns"))

    # Load checkpoint which contains model state
    checkpoint = torch.load(checkpoint_path)
    
    # Load config
    with open(config_path, 'r') as f:
        best_config = json.load(f)

    # Initialize model with best config
    model = MultiStockPredictorWithConfidence(
        num_stocks=best_config["num_stocks"],
        num_decoder_layers=best_config["num_decoder_layers"],
        num_heads=best_config["num_heads"],
        num_pred_layers=best_config["num_pred_layers"],
        attn_dropout=best_config["attn_dropout"],
        ff_dropout=best_config["ff_dropout"],
        encoder_name=best_config["encoder_name"]
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    
    # Get test results
    test_results = evaluate_model(model, test_dataset, device)
    print(f"Test Loss: {test_results['test_loss']:.4f}")
    
    # Run backtest
    backtest_results = run_backtest(
        test_results,
        returns_by_year,
        sp500_by_year
    )

if __name__ == "__main__":
    test_years = [2023]
    checkpoint_path = os.path.abspath("checkpoints/best_model/epoch0_batch19999.pt")
    config_path = os.path.abspath("checkpoints/best_model/params.json")
    data_dir = os.path.abspath("data")
    test_model(test_years, checkpoint_path, config_path, data_dir)