import torch
from torch.utils.data import DataLoader
import backtrader as bt
import pandas as pd
from collections import defaultdict
from train import collate_fn
from src.nscan.model import confidence_weighted_loss

def evaluate_model(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_predictions = []
    all_confidences = []
    all_returns = []
    all_dates = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            device_batch = {k: v.to(device) for k, v in batch.items()}
            
            predictions, confidences = model(
                input={
                    'input_ids': device_batch['input_ids'],
                    'attention_mask': device_batch['attention_mask']
                },
                stock_indices=device_batch['stock_indices']
            )
            
            loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
            total_loss += loss.item()
            
            # Store predictions and metadata
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())
            all_returns.append(device_batch['returns'].cpu())
            # You'll need to modify the dataset to include dates
            all_dates.extend([test_dataset.articles[i]['Date'].split()[0] 
                            for i in range(len(predictions))])
    
    return {
        'test_loss': total_loss / len(test_loader),
        'predictions': torch.cat(all_predictions),
        'confidences': torch.cat(all_confidences),
        'returns': torch.cat(all_returns),
        'dates': all_dates
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
                    stock_data = self.getdatabyname(str(stock))
                    self.buy(data=stock_data, size=position_size/stock_data.close[0])

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

def test_model():
    # Load your trained model
    model = MultiStockPredictorWithConfidence(...)
    model.load_state_dict(torch.load('best_model.pt'))
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