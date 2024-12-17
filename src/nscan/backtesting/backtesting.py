import backtrader as bt
from collections import defaultdict
import math
import pandas as pd

class NewsBasedStrategy(bt.Strategy):
    params = (
        ('confidence_threshold', 0.001),
        ('num_stocks', 10)
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
            
            print(f"\nDate: {current_date}")
            print(f"Portfolio Value: {self.broker.getvalue()}")
            # Combine predictions for the same day using confidence-weighted average
            weighted_predictions = defaultdict(list)
            weighted_confidences = defaultdict(list)
            
            for pred, conf, stocks in zip(predictions, confidences, self.stock_indices):
                for p, c, s in zip(pred, conf, stocks):
                    if s != 0 and c > self.p.confidence_threshold:  # Skip padding and low confidence predictions
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
            #print(sorted_stocks, flush=True)
            # Invest in top N stocks with highest predicted returns
            N = self.p.num_stocks  # Number of stocks to invest in
            total_value = self.broker.getvalue()
            position_size = total_value / N
            
            # Close existing positions
            for data in self.datas:
                if self.getposition(data).size != 0:
                    self.close(data)
            
            # Open new positions
            for stock_idx, pred in sorted_stocks[:N]:
                if pred > 0:  # Only go long if predicted return is positive
                    try:
                        if stock_idx == '-':
                            continue
                        idx = int(stock_idx)
                        stock_data = self.datas[idx]
                        if not math.isnan(stock_data.close[0]):
                            current_price = stock_data.close[0]
                            position_value = position_size/current_price
                            print(f"Buying stock {idx}: {position_value} units at price {current_price}")
                            self.buy(data=stock_data, 
                                size=position_value)
                    except Exception as e:
                        print(f"Error trading stock {stock}: {e}")

class ReturnsData(bt.feeds.PandasData):
    """Custom data feed that works with returns instead of prices"""
    lines = ('returns',)
    params = (
        ('returns', 'Returns'),  # Name of returns column in DataFrame
        ('openinterest', None),  # Not using these fields
        ('volume', None),
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
            self.lines.open[0] = self._price
            self.lines.high[0] = self._price
            self.lines.low[0] = self._price
            if math.isnan(self._price):
                print(f"WARNING: NaN price detected!")
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
            # Ensure index is datetime
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)

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
    
    # Print results with error handling
    strat = results[0]
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Handle Sharpe Ratio
    sharpe_analysis = strat.analyzers.sharperatio.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio', None)
    print(f'Sharpe Ratio: {f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A"}')
    
    # Handle Returns
    returns_analysis = strat.analyzers.returns.get_analysis()
    total_return = returns_analysis.get('rtot', None)
    print(f'Return: {total_return:.2%}' if total_return is not None else 'Return: N/A')
    
    # Handle Drawdown
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', None)
    print(f'Max Drawdown: {max_drawdown:.2%}' if max_drawdown is not None else 'Max Drawdown: N/A')
    
    return results
