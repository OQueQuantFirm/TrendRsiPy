import os
import ccxt
import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import csv
from dotenv import load_dotenv
from finta import TA

# Constants for CSV Field Names
TIMESTAMP = "Timestamp"
TRADING_SIGNAL = "Trading Signal"
PROPOSED_ENTRY_PRICE = "Proposed Entry Price"
ORDER_BOOK_IMBALANCE = "Order Book Imbalance"
RSI_FIELD = "RSI"
CSV_FIELD_NAMES = [TIMESTAMP, TRADING_SIGNAL, PROPOSED_ENTRY_PRICE, ORDER_BOOK_IMBALANCE, RSI_FIELD]

class RsiTrend:
    def __init__(self, symbol, leverage, amount, take_profit_percentage, stop_loss_percentage):
        self.symbol = symbol
        self.leverage = leverage
        self.amount = amount
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.trading_signals_df = self.load_trading_signals_from_csv("arb_rsi_trend.csv")
        self.trading_signals = []  
        self.logger = logging.getLogger(__name__)
        log_file_path = 'btc_rsi_trend.log'
        rotating_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=1024 * 1024,  # 1 MB per file
            backupCount=3  # Keep up to 3 backup log files
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Add the rotating handler to the root logger
        logging.getLogger().addHandler(rotating_handler)
        
        # Initialize the KuCoin Futures exchange instance
        self.exchange = ccxt.kucoinfutures({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('SECRET_KEY'),
            'password': os.getenv('PASSPHRASE'),
            'enableRateLimit': True  # Adjust as needed
        })

    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        try:
            if high_prices is None or low_prices is None or close_prices is None:
                print("Error: One or more data sources are None.")
                return None

            # Check if any of the input lists are empty
            if not high_prices.all() or not low_prices.all() or not close_prices.all():
                #print("Error: One or more data sources are empty.")
                return None

            # Calculate True Range (TR)
            tr = [max(hl, hc, lc) - min(hl, hc, lc) for hl, hc, lc in zip(high_prices, close_prices, low_prices)]

            # Calculate the Average True Range (ATR) using a period (e.g., 14)
            atr = np.mean(tr[-period:])
            return atr

        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return None 

    def calculate_rsi(self, close_prices, high_prices, low_prices, atr, period=14):
        try:
            # Create a DataFrame with required columns
            df = pd.DataFrame({'close': close_prices, 'open': close_prices, 'high': high_prices, 'low': low_prices})

            # Calculate the RSI using FinTa library
            df['rsi'] = TA.RSI(df, period=period)
            rsi = df['rsi'].iloc[-1]

            return rsi
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return None

    def calculate_smoothed_imbalance(self, data, alpha=0.1):
        try:
            #print("Input data:", data)
            #print("Alpha:", alpha)

            smoothed_data = [data[0]]  # Initialize with the first data value
            for i in range(1, len(data)):
                smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
                smoothed_data.append(smoothed_value)
                #print(f"Smoothed value at index {i}: {smoothed_value}")

            #print("Smoothed data:", smoothed_data)
            return smoothed_data

        except Exception as e:
            logging.error(f"Error calculating smoothed imbalance: {e}")
            return None

    def fetch_ohlcv_and_analyze_order_book(self, symbol, depth=100, max_retries=3):
        retries = 0
        # Initialize a list to store historical imbalance percentages
        historical_imbalance_percentage = []

        rsi = None
        current_imbalance_percentage = None
        close_prices = None
        high_prices = None
        low_prices = None
        bids = None
        asks = None

        while retries < max_retries:
            try:
                # Use time module to get the current timestamp
                current_time = int(time.time() * 1000)  # Convert seconds to milliseconds

                # Fetch OHLCV data for ATR and TR calculation
                ohlcv_data = self.exchange.fetch_ohlcv(symbol, '4h')  # Adjust timeframe as needed
                close_prices = np.array([item[4] for item in ohlcv_data])
                high_prices = np.array([item[2] for item in ohlcv_data])
                low_prices = np.array([item[3] for item in ohlcv_data])

                # Fetch volume data
                volume_data = np.array([item[5] for item in ohlcv_data])

                # Calculate True Range (TR)
                tr = [max(hl, hc, lc) - min(hl, hc, lc) for hl, hc, lc in zip(high_prices, close_prices, low_prices)]

                # Calculate Average True Range (ATR) using a period (e.g., 14)
                atr = np.mean(tr[-14:])

                rsi = self.calculate_rsi(close_prices, high_prices, low_prices, atr)

                # Fetch the order book for the specified symbol and depth
                order_book = self.exchange.fetch_order_book(symbol, limit=20)
                bids = order_book['bids']
                asks = order_book['asks']

                # Extract bid prices and quantities
                bid_prices = [bid[0] for bid in bids]
                bid_quantities = [bid[1] for bid in bids]

                # Extract ask prices and quantities
                ask_prices = [ask[0] for ask in asks]
                ask_quantities = [ask[1] for ask in asks]

                # Calculate the total volume of bids and asks
                total_bids_volume = sum(bid[1] for bid in bids)
                total_asks_volume = sum(ask[1] for ask in asks)

                # Calculate the current order book imbalance percentage
                current_imbalance_percentage = (
                    (total_bids_volume - total_asks_volume) / (total_bids_volume + total_asks_volume)
                ) * 100

                # Log order book analysis results
                self.logger.info(
                    f"Order Book Analysis for {symbol} - Imbalance: {current_imbalance_percentage:.2f}% - RSI: {rsi:.2f} - "
                    f"Current Market Price: {close_prices[-1]:.8f}"  # Print the current market price
                )

                # Print order book analysis results directly to the console
                print(
                    f"Order Book Analysis for {symbol} - Imbalance: {current_imbalance_percentage:.2f}% - RSI: {rsi:.2f} - "
                    f"Current Market Price: {close_prices[-1]:.8f}"  # Print the current market price
                )

                # Append the current imbalance percentage to the historical list
                historical_imbalance_percentage.append(current_imbalance_percentage)

                # Calculate smoothed order book imbalance using EMA
                smoothed_imbalance = self.calculate_smoothed_imbalance(historical_imbalance_percentage)

                # Generate trading signal and proposed entry price based on RSI and order book imbalance
                trading_signal, proposed_entry_price, take_profit_price, stop_loss_price = self.generate_trading_signal(
                    rsi,
                    current_imbalance_percentage,
                    close_prices,
                    high_prices,
                    low_prices,
                    bids,
                    asks
                )

                print("Trading Signal:", trading_signal)
                if proposed_entry_price:
                    print("Proposed Entry Price:", proposed_entry_price)
                    print("Take Profit Price:", take_profit_price)
                    print("Stop Loss Price:", stop_loss_price)

                # Exit the retry loop if data is successfully fetched and analyzed
                break

            except Exception as e:
                retries += 1
                self.logger.error(
                    f"Error fetching or analyzing order book: {e}" if e is not None else "Unknown error occurred.",
                    exc_info=True  # Include exception information in the log
                )
                self.logger.info(f"Retrying... ({retries}/{max_retries})")
                time.sleep(10)  # Wait for 10 seconds before retrying

        # Return the calculated values
        return rsi, current_imbalance_percentage, close_prices, high_prices, low_prices, bids, asks

    def calculate_take_profit_and_stop_loss(self, entry_price, leverage, take_profit_percentage, stop_loss_percentage):
        # Calculate the leverage-adjusted entry price
        leverage_adjusted_entry_price = entry_price / leverage

        # Calculate take profit and stop loss prices based on the leverage-adjusted entry price
        take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PERCENTAGE / 100), 8)
        stop_loss_price = round(entry_price * (1 - STOP_LOSS_PERCENTAGE / 100), 8)

        return take_profit_price, stop_loss_price

    def generate_trading_signal(self, rsi, imbalance_percentage, close_prices, high_prices, low_prices, bids, asks):
        try:
            self.logger.debug("Starting generate_trading_signal...")

            if imbalance_percentage >= 20:  # Positive imbalance condition
                self.logger.debug("Positive imbalance condition detected.")
                # Check for bullish RSI divergence (oversold RSI)
                if rsi < 28:
                    print("Bullish RSI divergence detected.")
                    proposed_entry_price = bids[0][0]

                    # Calculate take profit and stop loss prices
                    take_profit_price, stop_loss_price = self.calculate_take_profit_and_stop_loss(
                        proposed_entry_price,
                        self.leverage,
                        self.take_profit_percentage,
                        self.stop_loss_percentage
                    )

                    print("Validated Bullish Divergence (Long)")
                    return "Validated Bullish Divergence (Long)", proposed_entry_price, take_profit_price, stop_loss_price
                else:
                    #print("No bullish RSI divergence.")
                    return "No Entry", None, None, None
            elif imbalance_percentage <= -20:  # Negative imbalance condition
                self.logger.debug("Negative imbalance condition detected.")
                # Check for bearish RSI divergence (overbought RSI)
                if rsi > 72:
                    print("Bearish RSI divergence detected.")
                    proposed_entry_price = asks[0][0]

                    # Calculate take profit and stop loss prices
                    take_profit_price, stop_loss_price = self.calculate_take_profit_and_stop_loss(
                        proposed_entry_price,
                        self.leverage,
                        self.take_profit_percentage,
                        self.stop_loss_percentage
                    )

                    print("Validated Bearish Divergence (Short)")
                    return "Validated Bearish Divergence (Short)", proposed_entry_price, take_profit_price, stop_loss_price
                else:
                    #print("No bearish RSI divergence.")
                    return "No Entry", None, None, None

            self.logger.debug("Exiting generate_trading_signal...")

        except Exception as e:
            self.logger.error(f"Error in generate_trading_signal: {e}")
            self.logger.debug("Error in generate_trading_signal:", e)
            return "No Entry", None, None, None
        
    def create_order_with_percentage_levels(self, side, entry_price):
        try:
            print("Creating orders with percentage-based levels...")

            # Calculate take-profit and stop-loss prices
            take_profit_price, stop_loss_price = self.calculate_take_profit_and_stop_loss(
                entry_price,
                self.leverage,
                self.take_profit_percentage,  
                self.stop_loss_percentage  
            )

            print(f"Entry Price: {entry_price}")
            print(f"Take-Profit Price: {take_profit_price}")
            print(f"Stop-Loss Price: {stop_loss_price}")

            
            main_order = self.exchange.create_order(
                symbol=self.symbol,  
                type='limit',
                side=side,
                amount=self.amount,  
                price=entry_price,
                params={
                    'postOnly': True,
                    'timeInForce': 'GTC',
                    'leverage': self.leverage  
                }
            )
            print("Main Order Created:", main_order)

            # Create the stop-loss order
            stop_loss_order = self.exchange.create_order(
                symbol=self.symbol,  # Use self.symbol
                type='limit',
                side='sell' if side == 'buy' else 'buy',
                amount=self.amount,  # Use self.amount
                price=stop_loss_price
            )
            print("Stop-Loss Order Created:", stop_loss_order)

            # Create the take-profit order
            take_profit_order = self.exchange.create_order(
                symbol=self.symbol,  # Use self.symbol
                type='limit',
                side='sell' if side == 'buy' else 'buy',
                amount=self.amount,  # Use self.amount
                price=take_profit_price
            )
            print("Take-Profit Order Created:", take_profit_order)

            return main_order, stop_loss_order, take_profit_order

        except Exception as e:
            print(f"Error creating orders with percentage-based levels: {e}")
            return None, None, None
        
    def save_trading_signals_to_csv(self):
        try:
            file_path = "btc_rsi_trend.csv"
            # Check if the file already exists
            file_exists = os.path.exists(file_path)
            # Open the file in append mode
            with open(file_path, "a", newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELD_NAMES)
                # Write header only if the file is newly created
                if not file_exists:
                    print("Writing header to CSV file...")
                    csv_writer.writeheader()
                # Write the data to the CSV file
                for signal in self.trading_signals_df.to_dict(orient='records'):
                    #print(f"Writing signal to CSV: {signal}")
                    csv_writer.writerow({
                        TIMESTAMP: signal[TIMESTAMP],
                        TRADING_SIGNAL: signal[TRADING_SIGNAL],
                        PROPOSED_ENTRY_PRICE: signal[PROPOSED_ENTRY_PRICE],
                        ORDER_BOOK_IMBALANCE: signal[ORDER_BOOK_IMBALANCE],
                        RSI_FIELD: signal[RSI_FIELD]
                    })
            print("CSV file saved successfully.")
        except Exception as e:
            print(f"Error saving trading signals to CSV: {e}")
                
    def load_trading_signals_from_csv(self, file_path):
        try:
            #print("Reading CSV file:", file_path)
            historical_signals = pd.read_csv(file_path, parse_dates=["Timestamp"], na_values=['nan', 'NaN'], dtype={'Trading Signal': str})
            # Print unique values in the "Trading Signal" column
            #print("Unique values in 'Trading Signal' column:", historical_signals["Trading Signal"].unique())
            # Replace NaN values in 'Trading Signal' column with 'No Entry'
            #print("Replacing NaN values in 'Trading Signal' column with 'No Entry'...")
            historical_signals.fillna(value={'Trading Signal': 'No Entry'}, inplace=True)
            with open(file_path, "r", newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                # Check if the required columns exist in the CSV file
                required_columns = {"Timestamp", "Trading Signal", "Proposed Entry Price", "Order Book Imbalance", "RSI"}
                if not required_columns.issubset(csv_reader.fieldnames):
                    #print(f"Error: The CSV file is missing one or more required columns. Actual columns: {csv_reader.fieldnames}")
                    return historical_signals
                signals_list = []  # Create a list to store signals
                for row in csv_reader:
                    timestamp_str = row["Timestamp"]
                    timestamp = pd.to_datetime(timestamp_str) if timestamp_str != 'NaT' else pd.NaT
                    trading_signal = row["Trading Signal"]
                    proposed_entry_price = float(row["Proposed Entry Price"]) if row["Proposed Entry Price"] else None
                    order_book_imbalance = float(row["Order Book Imbalance"]) if row["Order Book Imbalance"] else None
                    rsi = float(row["RSI"]) if row["RSI"] else None
                    signal = {
                        "timestamp": timestamp,
                        "trading_signal": trading_signal,
                        "proposed_entry_price": proposed_entry_price,
                        "order_book_imbalance": order_book_imbalance,
                        "rsi": rsi
                    }
                    signals_list.append(signal)  # Append each signal to the list
                # Print DataFrame columns here
                #print("Columns in historical_signals DataFrame:", historical_signals.columns)
                historical_signals = pd.DataFrame(columns=["Timestamp", "Trading Signal", "Proposed Entry Price", "Order Book Imbalance", "RSI"])

        except FileNotFoundError:
            # The file may not exist initially, which is fine
            print("CSV file not found. Returning empty DataFrame.")
            historical_signals = pd.DataFrame(columns=CSV_FIELD_NAMES)
            return historical_signals
        except Exception as e:
            print(f"Error loading trading signals from CSV: {e}")
            return None

        return historical_signals

    def execute_order_book_analysis(self):
        while True:
            try:
                self.logger.info("Fetching OHLCV data and analyzing order book...")
                rsi, imbalance_percentage, close_prices, high_prices, low_prices, bids, asks = self.fetch_ohlcv_and_analyze_order_book(symbol_to_analyze)
                # Generate trading signal and proposed entry price based on RSI and order book imbalance
                trading_signal, proposed_entry_price, take_profit_price, stop_loss_price = self.generate_trading_signal(
                    rsi,
                    imbalance_percentage,
                    close_prices,
                    high_prices,
                    low_prices,
                    bids,
                    asks
                )
                # Get the current timestamp
                timestamp = int(time.time() * 1000)
                timestamp_datetime = datetime.fromtimestamp(timestamp / 1000.0)
                # Create a new signal dictionary
                new_signal = {
                    TIMESTAMP: timestamp_datetime,
                    TRADING_SIGNAL: trading_signal,
                    PROPOSED_ENTRY_PRICE: proposed_entry_price,
                    ORDER_BOOK_IMBALANCE: imbalance_percentage,  # Include order book imbalance
                    RSI_FIELD: rsi  # Include RSI
                }
                
                # Print the relevant information
                self.logger.debug(f"Timestamp: {timestamp_datetime}")
                #print(f"Trading Signal: {trading_signal}")
                if proposed_entry_price:
                    print(f"Proposed Entry Price: {proposed_entry_price}")
                # Append the new signal to trading_signals_df
                self.trading_signals_df = pd.concat([self.trading_signals_df, pd.DataFrame([new_signal])], ignore_index=True)
                # Call the saving function to update the CSV file
                self.save_trading_signals_to_csv()

                if trading_signal != "No Entry" and proposed_entry_price:
                    # Create a limit order based on the signal
                    if trading_signal.startswith("Validated Bullish"):
                        # Pass the desired take profit and stop loss percentages to the order creation function
                        self.create_order_with_percentage_levels('buy', proposed_entry_price)
                    elif trading_signal.startswith("Validated Bearish"):
                        # Pass the desired take profit and stop loss percentages to the order creation function
                        self.create_order_with_percentage_levels('sell', proposed_entry_price)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
            print("Exiting execute_order_book_analysis...")
            print("=" * 50)
            time.sleep(60)

load_dotenv()
print("Environment variables loaded successfully.")
logging.basicConfig(filename='btc_rsi_trend.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
print("Logging initiated successfully.")

symbol_to_analyze = 'BTC/USDT:USDT' 
leverage = 10  
amount = 1  
TAKE_PROFIT_PERCENTAGE = 1.35  
STOP_LOSS_PERCENTAGE = 1.35  

analyzer = RsiTrend(symbol_to_analyze, leverage, amount, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE)
historical_trading_signals = analyzer.load_trading_signals_from_csv("btc_rsi_trend.csv")
trading_signals = historical_trading_signals if not historical_trading_signals.empty else []
analyzer.execute_order_book_analysis()
analyzer.save_trading_signals_to_csv()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)
