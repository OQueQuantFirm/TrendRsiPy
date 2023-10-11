# TrendRsiPy
Profitable Trading Strategy for Kucoin Futures using Order Book Imbalance and RSI and create a limit order with TP and SL

## Overview of the Trading Strategy

The RSI (Relative Strength Index) trend strategy is a trading approach that combines technical analysis indicators, specifically the RSI, with order book analysis. The strategy aims to identify potential trend reversals or continuations in the price of an asset and generate trading signals accordingly.

  #### 1. **Relative Strength Index (RSI):**
     - The RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is typically used to identify overbought or oversold conditions in a market. Traditional RSI signals include:
        - **Overbought (RSI > 70):** Indicates a potential sell signal as the asset may be overvalued, and a reversal might occur.
        - **Oversold (RSI < 30):** Indicates a potential buy signal as the asset may be undervalued, and a reversal might occur.
  
  #### 2. **Order Book Analysis:**
     - Order book analysis involves studying the current bids and asks in the market. It provides insights into the supply and demand for an asset at different price levels.
     - The strategy incorporates order book imbalance, which is the difference between the total volume of bids and asks. A positive imbalance suggests more buying interest, while a negative imbalance suggests more selling interest.
  
  ### Combined Strategy:
  
  1. **RSI Analysis:**
     - The strategy monitors the RSI of the asset over a specified period (e.g., 14 periods).
     - Bullish Divergence: If the RSI is below a certain threshold (e.g., 28) during a positive order book imbalance, it may indicate a bullish divergence, suggesting a potential buying opportunity.
     - Bearish Divergence: If the RSI is above a certain threshold (e.g., 72) during a negative order book imbalance, it may indicate a bearish divergence, suggesting a potential selling opportunity.
  
  2. **Order Book Analysis:**
     - The strategy calculates the order book imbalance percentage based on the total volume of bids and asks.
     - A positive imbalance indicates more buying interest, while a negative imbalance indicates more selling interest.
  
  3. **Trading Signals:**
     - Based on the combined analysis of RSI and order book imbalance, the strategy generates trading signals:
        - **Validated Bullish Divergence (Long):** Indicates a potential buying opportunity with a proposed entry price derived from the order book (e.g., highest bid).
        - **Validated Bearish Divergence (Short):** Indicates a potential selling opportunity with a proposed entry price derived from the order book (e.g., lowest ask).
        - **No Entry:** If no favorable conditions are met.
  
  4. **Risk Management:**
     - The strategy incorporates risk management by setting take-profit and stop-loss levels based on a percentage of the entry price. This helps limit potential losses and secure profits.
  
  5. **Continuous Monitoring:**
     - The strategy continuously monitors the market conditions, periodically fetching and analyzing OHLCV data and order book information.
     - Trading signals, along with relevant information, are logged and saved to a CSV file for record-keeping.
  
  This combined RSI trend strategy aims to provide a systematic approach to trading by considering both momentum (RSI) and market depth (order book analysis). It allows for automation and continuous monitoring, enabling timely execution of trades based on predefined criteria.

2. ### Setting up the Project:

   2.1. **Clone the Repository:**
      - Open a command prompt or Git Bash.
      - Navigate to the directory where you want to clone the repository.
      - Run the following command:
        ```bash
        git clone https://github.com/OQueQuantFirm/TrendRsiPy.git
        ```
   
   2.2 **Install Required Libraries:**
      - Navigate to the project directory:
        ```bash
        cd TrendRsiPy
        ```
      - Install the required Python libraries:
        ```bash
        pip install ccxt numpy pandas finta python-dotenv
        ```
   
   2.3 **Create Environment Variables:**
      - Rename example.env to `.env` in the project directory.
      - Add the following lines to the `.env` file, replacing placeholders with your KuCoin Futures API credentials:
        ```plaintext
        API_KEY=your_api_key
        SECRET_KEY=your_secret_key
        PASSPHRASE=your_passphrase
        ```
3. **Code Overview**
   
   3.1 **Constants and CSV Field Names**
      - `TIMESTAMP`, `TRADING_SIGNAL`, `PROPOSED_ENTRY_PRICE`, `ORDER_BOOK_IMBALANCE`, and `RSI_FIELD` are constants representing the field names in the CSV file.
   
      ```python
      TIMESTAMP = "Timestamp"
      TRADING_SIGNAL = "Trading Signal"
      PROPOSED_ENTRY_PRICE = "Proposed Entry Price"
      ORDER_BOOK_IMBALANCE = "Order Book Imbalance"
      RSI_FIELD = "RSI"
      CSV_FIELD_NAMES = [TIMESTAMP, TRADING_SIGNAL, PROPOSED_ENTRY_PRICE, ORDER_BOOK_IMBALANCE, RSI_FIELD]
      ```
   
   3.2 **RsiTrend Class**
      - This class represents the main trading strategy.
      - It has methods for initializing the strategy, calculating indicators, generating trading signals, executing orders, and saving/loading trading signals.
   
      ```python
      class RsiTrend:
          def __init__(self, symbol, leverage, amount, take_profit_percentage, stop_loss_percentage):
              # Initialization code...
   
          def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
              # ATR calculation code...
   
          def calculate_rsi(self, close_prices, high_prices, low_prices, atr, period=14):
              # RSI calculation code...
   
          # Other methods...
      ```
   
   3.3 **Order Book Analysis and Trading Signal Generation**
      - The `fetch_ohlcv_and_analyze_order_book` method fetches OHLCV data and analyzes the order book.
      - The `generate_trading_signal` method generates trading signals based on RSI and order book imbalance.
   
      ```python
      def fetch_ohlcv_and_analyze_order_book(self, symbol, depth=100, max_retries=3):
          # OHLCV and order book analysis code...
   
      def generate_trading_signal(self, rsi, imbalance_percentage, close_prices, high_prices, low_prices, bids, asks):
          # Trading signal generation code...
      ```
   
   3.4 **Order Execution**
      - The `create_order_with_percentage_levels` method creates main, stop-loss, and take-profit orders.
   
      ```python
      def create_order_with_percentage_levels(self, side, entry_price):
          # Order creation code...
      ```
   
    3.5 **Saving and Loading Trading Signals**
      - The `save_trading_signals_to_csv` and `load_trading_signals_from_csv` methods handle saving and loading trading signals to/from a CSV file.
   
      ```python
      def save_trading_signals_to_csv(self):
          # CSV saving code...
   
      def load_trading_signals_from_csv(self, file_path):
          # CSV loading code...
      ```
   
   3.6 **Main Execution Loop**
      - The `execute_order_book_analysis` method contains the main execution loop, fetching data, generating signals, and executing orders.
   
      ```python
      def execute_order_book_analysis(self):
          # Main execution loop code...
      ```
   
   3.7 **Environment Setup and Initialization**
      - Loading environment variables, setting up logging, and initializing the trading strategy.
   
      ```python
      load_dotenv()
      logging.basicConfig(filename='btc_rsi_trend.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
   
      symbol_to_analyze = 'BTC/USDT:USDT'
      leverage = 10
      amount = 1
      TAKE_PROFIT_PERCENTAGE = 1.35
      STOP_LOSS_PERCENTAGE = 1.35
   
      analyzer = RsiTrend(symbol_to_analyze, leverage, amount, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE)
      ```
      
4. **Adjustable Parameters**
    Users can adjust several parameters in the trading code based on their preferences. Here are the adjustable parameters along with their significance:
   
   4.1 **`symbol_to_analyze`**
      - Symbol or trading pair that the strategy will analyze. Users can change this to any trading pair available on the KuCoin Futures exchange.
   
      ```python
      symbol_to_analyze = 'BTC/USDT:USDT'
      ```
   
   4.2 **`leverage`**
      - Leverage used in trading. It determines the amount of borrowed capital. Higher leverage amplifies both potential profits and losses.
   
      ```python
      leverage = 10
      ```
   
   4.3 **`amount`**
      - The amount of the asset to trade. It represents the quantity of the asset in each trade.
   
      ```python
      amount = 1
      ```
   
   4.4 **`TAKE_PROFIT_PERCENTAGE`**
      - Percentage profit at which a take-profit order is triggered. For example, if set to 1.35, the take-profit order will be placed at 1.35% profit from the entry price.
   
      ```python
      TAKE_PROFIT_PERCENTAGE = 1.35
      ```
   
   4.5 **`STOP_LOSS_PERCENTAGE`**
      - Percentage loss at which a stop-loss order is triggered. For example, if set to 1.35, the stop-loss order will be placed at 1.35% loss from the entry price.
   
      ```python
      STOP_LOSS_PERCENTAGE = 1.35
      ```
   
5. **Understanding Order Book Analysis**
       Certainly! The `fetch_ohlcv_and_analyze_order_book` function is responsible for retrieving OHLCV (Open, High, Low, Close, Volume) data, calculating indicators such as RSI (Relative Strength Index) and ATR (Average True Range), and analyzing the order book for a given trading symbol. Let's break down the function:
   
   ```python
   def fetch_ohlcv_and_analyze_order_book(self, symbol, depth=100, max_retries=3):
       retries = 0
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
               # Fetch OHLCV data for ATR and TR calculation
               ohlcv_data = self.exchange.fetch_ohlcv(symbol, '4h')
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
                   f"Current Market Price: {close_prices[-1]:.8f}"
               )
   
               # Append the current imbalance percentage to the historical list
               historical_imbalance_percentage.append(current_imbalance_percentage)
   
               # Calculate smoothed order book imbalance using EMA
               smoothed_imbalance = self.calculate_smoothed_imbalance(historical_imbalance_percentage)
   
               # ... (continue with the rest of the function)
               
               # Return the calculated values
               return rsi, current_imbalance_percentage, close_prices, high_prices, low_prices, bids, asks
   
           except Exception as e:
               retries += 1
               self.logger.error(
                   f"Error fetching or analyzing order book: {e}" if e is not None else "Unknown error occurred.",
                   exc_info=True
               )
               self.logger.info(f"Retrying... ({retries}/{max_retries})")
               time.sleep(10)  # Wait for 10 seconds before retrying
   ```
   
   Here's an overview of the key steps in the function:
   
   5.1 **OHLCV Data Retrieval:**
      - The function uses `self.exchange.fetch_ohlcv` to retrieve historical OHLCV data for the specified symbol and timeframe (4 hours in this case).
      - Close, high, and low prices are extracted from the OHLCV data.
   
   5.2 **True Range (TR) Calculation:**
      - The True Range (TR) is calculated based on the high, low, and close prices.
   
   5.3 **Average True Range (ATR) Calculation:**
      - ATR is calculated as the average of the True Range values over a specified period (14 in this case).
   
   5.4 **RSI Calculation:**
      - RSI is calculated using the `calculate_rsi` method from the FinTa library.
   
   5.5 **Order Book Retrieval:**
      - The order book is fetched using `self.exchange.fetch_order_book` for the specified symbol with a limit of 20 bids and asks.
   
   5.6 **Order Book Analysis:**
      - Bid and ask prices and quantities are extracted from the order book.
      - Total volume for bids and asks is calculated.
      - Current order book imbalance percentage is calculated based on bid and ask volumes.
      - Results, including imbalance percentage, RSI, and current market price, are logged.
   
   5.7 **Retry Mechanism:**
      - The function includes a retry mechanism to handle potential errors during data retrieval or analysis.
   
   5.8 **Return Values:**
      - The function returns the calculated values, including RSI, current imbalance percentage, OHLCV data, and order book data.
   
6. **RSI and Divergence Detection**
       Certainly! Let's explore the `calculate_rsi` function and understand how it calculates the Relative Strength Index (RSI), as well as the conditions for detecting bullish and bearish RSI divergences:
   
   ```python
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
   ```
   
   **RSI Calculation:**
   6.1 A DataFrame (`df`) is created using the input close prices (`close_prices`) and OHLC data (`high_prices`, `low_prices`). The 'open' column is set equal to the 'close' column.
   
   6.2 The RSI is calculated using the `TA.RSI` function from the FinTa library. The resulting RSI values are added as a new column ('rsi') to the DataFrame.
   
   6.3 The function returns the last (latest) RSI value.
   
   **Bullish RSI Divergence:**
   - A bullish RSI divergence is detected when the RSI is less than 28.
   - The specific condition is:
     ```python
     if rsi < 28:
         print("Bullish RSI divergence detected.")
         proposed_entry_price = bids[0][0]
         # Additional code for creating orders and calculating take profit and stop loss prices.
     ```
   
     In this case, if the RSI is below 28, it indicates an oversold condition, and a potential bullish divergence is identified. The strategy then considers creating orders for a long position.
   
   **Bearish RSI Divergence:**
   - A bearish RSI divergence is detected when the RSI is greater than 72.
   - The specific condition is:
     ```python
     if rsi > 72:
         print("Bearish RSI divergence detected.")
         proposed_entry_price = asks[0][0]
         # Additional code for creating orders and calculating take profit and stop loss prices.
     ```
   
     If the RSI is above 72, it suggests an overbought condition, and a potential bearish divergence is identified. The strategy then considers creating orders for a short position.
   
   These conditions are based on traditional interpretations of RSI levels. RSI values below 30 typically indicate oversold conditions, suggesting potential buying opportunities (bullish divergence). Conversely, RSI values above 70 often suggest overbought conditions, indicating potential selling opportunities (bearish divergence). The specific thresholds (28 and 72 in this case) can be adjusted based on the trader's preferences and the characteristics of the market being traded.
   
8. **Creating Trading Signals**
    This function essentially defines the logic for generating trading signals based on a combination of order book imbalance and RSI conditions. If the strategy identifies specific conditions indicative of a potential trend reversal (bullish or bearish), it generates corresponding trading signals for long or short positions, along with proposed entry prices, take-profit prices, and stop-loss prices. Adjustments to the RSI thresholds and imbalance conditions can be made based on the trader's preferences and market characteristics.
   
   ```python
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
                   return "No Entry", None, None, None
   
           self.logger.debug("Exiting generate_trading_signal...")
   
       except Exception as e:
           self.logger.error(f"Error in generate_trading_signal: {e}")
           self.logger.debug("Error in generate_trading_signal:", e)
           return "No Entry", None, None, None
   ```
   
   **Signal Generation Logic:**
   - The function starts by checking whether the order book imbalance percentage is greater than or equal to 20 (positive imbalance) or less than or equal to -20 (negative imbalance).
   
   **Positive Imbalance Condition:**
   - If the condition is met, it checks for a bullish RSI divergence (oversold RSI) by verifying if the RSI is less than 28.
     ```python
     if imbalance_percentage >= 20:
         if rsi < 28:
             # Additional code for creating orders and calculating take profit and stop loss prices.
             return "Validated Bullish Divergence (Long)", proposed_entry_price, take_profit_price, stop_loss_price
     ```
   
   **Negative Imbalance Condition:**
   - If the condition is met, it checks for a bearish RSI divergence (overbought RSI) by verifying if the RSI is greater than 72.
     ```python
     elif imbalance_percentage <= -20:
         if rsi > 72:
             # Additional code for creating orders and calculating take profit and stop loss prices.
             return "Validated Bearish Divergence (Short)", proposed_entry_price, take_profit_price, stop_loss_price
     ```
   
   **No Entry Condition:**
   - If none of the above conditions are met, it returns "No Entry" as the trading signal.
     ```python
     else:
         return "No Entry", None, None, None
     ```
      
9. **Executing Trades**
    This function automates the process of creating three related orders (main, stop-loss, and take-profit) based on calculated price levels. It utilizes the exchange's API to execute these orders, and the print statements provide transparency into the order creation process:

```python
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

        # Create the main order
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
            symbol=self.symbol,
            type='limit',
            side='sell' if side == 'buy' else 'buy',
            amount=self.amount,
            price=stop_loss_price
        )
        print("Stop-Loss Order Created:", stop_loss_order)

        # Create the take-profit order
        take_profit_order = self.exchange.create_order(
            symbol=self.symbol,
            type='limit',
            side='sell' if side == 'buy' else 'buy',
            amount=self.amount,
            price=take_profit_price
        )
        print("Take-Profit Order Created:", take_profit_order)

        return main_order, stop_loss_order, take_profit_order

    except Exception as e:
        print(f"Error creating orders with percentage-based levels: {e}")
        return None, None, None
```

   **Order Creation Process:**
   9.1 **Calculate Prices:**
      - The function calculates the take-profit and stop-loss prices based on the entry price, leverage, and specified percentage levels (`self.take_profit_percentage` and `self.stop_loss_percentage`).
   
   9.2 **Print Prices:**
      - The calculated entry price, take-profit price, and stop-loss price are printed to the console for reference.
   
   9.3 **Create Main Order:**
      - The main order is created using the `create_order` method of the exchange instance (`self.exchange`).
      - Parameters such as `symbol`, `type` (limit order), `side` (buy/sell), `amount`, `price` (entry price), and additional parameters like `postOnly`, `timeInForce`, and `leverage` are specified.
      - The main order is printed to the console for confirmation.
   
   9.4 **Create Stop-Loss Order:**
      - A stop-loss order is created using the same `create_order` method with adjustments for the opposite side (sell if the main order is buy, and vice versa).
      - The stop-loss order is printed to the console for confirmation.
   
   9.5 **Create Take-Profit Order:**
      - Similarly, a take-profit order is created using the same method.
      - The take-profit order is printed to the console for confirmation.
   
   9.6 **Return Orders:**
      - The created main, stop-loss, and take-profit orders are returned as a tuple.
   
   **Note:** Error handling is implemented to catch any exceptions that may occur during the order creation process.


10. **Saving and Loading Trading Signals**
    
   These functions facilitate the persistence of trading signals, allowing the program to save signals to a CSV file and load historical signals from the same file. The CSV file serves as a record of past trading signals with relevant information such as timestamp, trading signal, proposed entry price, order book imbalance, and RSI.

```python
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
        historical_signals = pd.read_csv(file_path, parse_dates=["Timestamp"], na_values=['nan', 'NaN'], dtype={'Trading Signal': str})
        # Replace NaN values in 'Trading Signal' column with 'No Entry'
        historical_signals.fillna(value={'Trading Signal': 'No Entry'}, inplace=True)
        with open(file_path, "r", newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            # Check if the required columns exist in the CSV file
            required_columns = {"Timestamp", "Trading Signal", "Proposed Entry Price", "Order Book Imbalance", "RSI"}
            if not required_columns.issubset(csv_reader.fieldnames):
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

            historical_signals = pd.DataFrame(signals_list)  # Convert the list to a DataFrame
    except FileNotFoundError:
        print("CSV file not found. Returning empty DataFrame.")
        historical_signals = pd.DataFrame(columns=CSV_FIELD_NAMES)
        return historical_signals
    except Exception as e:
        print(f"Error loading trading signals from CSV: {e}")
        return None

    return historical_signals
```

**`save_trading_signals_to_csv` Function:**
- This function saves the trading signals to a CSV file (`btc_rsi_trend.csv`).
- It checks whether the file already exists and opens it in append mode.
- If the file is newly created, it writes the header with field names.
- It then iterates over the trading signals stored in `self.trading_signals_df` and writes each signal as a row in the CSV file.

**`load_trading_signals_from_csv` Function:**
- This function loads historical trading signals from the CSV file specified by the `file_path`.
- It uses `pd.read_csv` to read the CSV file into a DataFrame (`historical_signals`).
- NaN values in the "Trading Signal" column are replaced with 'No Entry'.
- It reads the CSV file using `csv.DictReader` to check for required columns and then constructs a list of signals.
- The list is converted into a DataFrame (`historical_signals`).

11. **Executing the Trading Loop**
    This function forms the core of the trading algorithm, continuously fetching data, analyzing the order book, generating signals, and executing trades based on predefined conditions. The loop runs indefinitely, making it a key component for automated trading:
    
```python
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
```

**Explanation:**
- The `execute_order_book_analysis` function contains an infinite loop (`while True`) to continuously analyze the order book.
- Inside the loop, it fetches OHLCV data and performs order book analysis using the `fetch_ohlcv_and_analyze_order_book` function.
- It generates trading signals and proposed entry prices based on RSI and order book imbalance using the `generate_trading_signal` function.
- It gets the current timestamp and creates a new signal dictionary (`new_signal`) containing relevant information.
- The information is printed to the console and appended to `self.trading_signals_df`, which stores historical trading signals.
- The `save_trading_signals_to_csv` function is called to update the CSV file with the new signal.
- If a valid trading signal (`"Validated Bullish"` or `"Validated Bearish"`) is generated and a proposed entry price exists, it calls the `create_order_with_percentage_levels` function to execute orders.
- The loop then waits for 60 seconds (`time.sleep(60)`) before starting the next iteration.


12. **Running the Trading Bot**
    By following these steps, you should be able to set up and run the trading bot in a Jupyter Notebook environment on Windows.
    
   ### Step 1: Set Up Jupyter Notebook
   1. Ensure that Python is installed on your system. You can download and install Python from [python.org](https://www.python.org/downloads/).
   2. Open a command prompt and install Jupyter Notebook using the following command:
      ```bash
      pip install jupyter
      ```
   
   ### Step 2: Install Required Libraries
   Ensure that you have all the required libraries installed. You can install them using the following command:
   ```bash
   pip install ccxt pandas finta python-dotenv
   ```
   
   ### Step 3: Prepare the Trading Bot Script
   Copy the entire trading bot script, including the import statements, class definition, and main execution code, into a Jupyter Notebook cell.
   
   ### Step 4: Create a `.env` File
   Create a file named `.env` in the same directory as your Jupyter Notebook. Add the following lines to the `.env` file, replacing placeholder values with your KuCoin API credentials:
   ```env
   API_KEY=your_api_key
   SECRET_KEY=your_secret_key
   PASSPHRASE=your_passphrase
   ```
   
   ### Step 5: Run the Jupyter Notebook
   1. Open a command prompt and navigate to the directory where your Jupyter Notebook is located.
   2. Start Jupyter Notebook by running the following command:
      ```bash
      jupyter notebook
      ```
   3. Your default web browser should open with the Jupyter Notebook interface.
   4. Navigate to the Jupyter Notebook containing the trading bot script and open it.
   
   ### Step 6: Execute the Trading Bot Script
   Run the cells in the Jupyter Notebook sequentially by clicking the "Run" button or using the Shift + Enter keyboard shortcut. Ensure that each cell executes without errors.
   
   ### Additional Considerations:
   - **Environment Variables:** Double-check that your KuCoin API credentials are correctly set in the `.env` file.
   - **API Permissions:** Make sure that your API key has the necessary permissions for trading on KuCoin.
   - **Security:** Do not share your API credentials publicly, and keep the `.env` file secure.
   
   ### Warnings:
   - **Real Trading:** Exercise caution when running trading bots in a real trading environment. Start with a paper trading account or use a small amount of funds to test the bot's behavior.
   - **Review the Code:** Understand the code thoroughly before running it, especially if it involves trading with real funds.
   - **Monitor the Bot:** Regularly monitor the bot's performance and be prepared to intervene if needed.

### Conclusion
Traders are encouraged to experiment with the code and parameters to adapt the trading bot to their preferences and strategies. Understanding the code thoroughly is crucial before engaging in real trading activities. Additionally, users should start with paper trading or a small amount of funds to test the bot's behavior in a controlled environment.

By exploring and customizing the code, users can gain insights into algorithmic trading strategies and potentially develop more sophisticated bots. Remember to stay informed about market conditions, regularly monitor the bot's performance, and be cautious when dealing with real funds.

### Additional Resources
Certainly! Here are some additional resources for further learning:

1. **ccxt Library Documentation:**
   - [ccxt Library Documentation](https://ccxt.readthedocs.io/) - Explore the documentation for the ccxt library, which provides a unified way to access cryptocurrency trading data across various exchanges.

2. **FinTA Library Documentation:**
   - [FinTA Library Documentation](https://finta.readthedocs.io/) - Refer to the FinTA library documentation for details on technical analysis indicators, including the Relative Strength Index (RSI).

3. **KuCoin Futures API Documentation:**
   - [KuCoin Futures API Documentation](https://docs.kucoin.com/futures/) - If you plan to use the KuCoin exchange, refer to their API documentation for information on accessing trading features.

4. **Algorithmic Trading and Strategies:**
   - [Algorithmic Trading Strategies for Cryptocurrencies](https://www.investopedia.com/terms/a/algorithmictrading.asp) - Investopedia provides an overview of algorithmic trading strategies, including those applicable to cryptocurrencies.

5. **Cryptocurrency Trading Tips:**
   - [Cryptocurrency Trading Tips for Beginners](https://cointelegraph.com/bitcoin-for-beginners/what-are-cryptocurrencies) - Cointelegraph offers tips for beginners in cryptocurrency trading.

6. **Risk Management in Trading:**
   - [Risk Management in Trading](https://www.investopedia.com/trading/how-to-create-trading-plan/) - Learn about the importance of risk management in trading and how to create a trading plan.

7. **Paper Trading:**
   - [What is Paper Trading?](https://www.investopedia.com/terms/p/papertrading.asp) - Understand the concept of paper trading, a risk-free way to practice trading strategies.

8. **Quantitative Trading Strategies:**
   - [Quantitative Trading Strategies: An Introduction](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp) - Explore the basics of quantitative trading strategies.


# RSI Trend Analysis with Order Book Imbalance

This guide explains how to use the provided Python code to perform Relative Strength Index (RSI) trend analysis with order book imbalance on the KuCoin Futures exchange. The code is designed to analyze the order book, calculate RSI, and generate trading signals based on certain conditions. Additionally, it allows you to customize symbols and timeframes for analysis.

### Code Customization

Open the provided Python script (`BTCUSDT_V1.0.ipynb`) and follow the instructions below to customize the code:

# Call execute_order_book_analysis directly for your single symbol analysis, the code is located at the bottom

   ```python
   symbol_to_analyze = 'BTC/USDT:USDT'
   leverage = 10  # Define your desired leverage
   amount = 1  # Define your desired amount
   execute_order_book_analysis(symbol_to_analyze, leverage, amount)
   ```

1. **Symbol and Timeframe**: Set the `symbol_to_analyze` variable to the trading pair you want to analyze, and adjust the timeframe by modifying the `ohlcv_data = self.exchange.fetch_ohlcv(symbol, '15m')` line.

   ```python
   symbol_to_analyze = 'YOUR_SYMBOL_HERE'  # Example: 'BTC/USDT:USDT'
   ```

2. **Leverage and Amount**: Set the leverage and trading amount according to your preferences.

   ```python
   leverage = 10  # Example: 10X leverage
   amount = 5   # Example: 5 units of trading token
   ```

3. **Take Profit and Stop Loss Percentages**: Define the take profit and stop loss percentages.

   ```python
   TAKE_PROFIT_PERCENTAGE = 1.35   # Example: 13.5%
   STOP_LOSS_PERCENTAGE = 1.35    # Example: 13.5%
   ```

4. **Logging and CSV File**: Customize the logging configuration and CSV file settings if needed.

   ```python
   log_file_path = 'btc_rsi_trend.log'  # Specify the log file path by changing the symbol name
   ```

   ```python
   file_path = "btc_rsi_trend.csv"  # Specify the CSV file path by changing the symbol name
   ```

The script will continuously fetch OHLCV data, analyze the order book, and generate trading signals based on RSI and order book imbalance.

## Additional Notes

- The code utilizes the [CCXT](https://github.com/ccxt/ccxt) library for interacting with the KuCoin Futures API.

- The script is designed to run in a loop, fetching and analyzing data every 2 minutes (adjustable in the `time.sleep(120)` line).

Feel free to experiment with different symbols, timeframes, and parameters to adapt the code to your trading preferences.

### Notes:
- Ensure your API keys have the necessary permissions and are funded for trading.
- Always be cautious when running trading programs, especially with real funds.
- Periodically check for updates to libraries and the repository.


