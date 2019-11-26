"""
Live Trading with IB Modular Design
Version 1.0.1
Platform: IB API and TWS
By: Zach Oakes


Revision Notes:
1.0.0 (07/30/2019) - Initial
1.0.1 (08/06/2019) - Add fixed stop and trailing stop logic
1.0.2 (11/4/2019) - Added EOD exit, reset time for TS/CS check to constant
1.0.3 (11.4.19)   - Added MM -- FR
1.0.4 (11.6.19)  - Added Fxd Qty (AND DYNAMIC FXD)
"""
###############################################################################
# Import required libraries
import calendar
import datetime
from ib_insync import *
from ibapi import *
import logging
import numpy as np
import pandas as pd
import pytz
import sys
import time

###############################################################################
# Required variables for the algo
# Set to True when using Spyder
USING_NOTEBOOK = True

# Set timezone for your TWS setup
TWS_TIMEZONE = pytz.timezone('US/Central')

# Set fixed position size amount
QTY = 100
# Set value for stop loss
SL = 100.00
# Set value for profit required before trailing exit is turned on
PT = 100.00
# Set value for percent trailing stop once turned on
PCT = 0.15 # percent as decimal, 0.20 = 20%


'''MM Globals '''
POS_SIZE_USD = 10000
CLOSE_DICT = {
    'SPY': 300,
    'ROKU':150,
    'TSLA':300,
    'AMD':37,
    'INTC':57,
    'UAL':93
}

QTY_DICT = {}

PNL = {}
DELTA = 1000


###############################################################################
class IBAlgoStrategy(object):
    """
    Algorithmic Strategy for Interactive Brokers.
    """
    def __init__(self):
        """Initialize algo"""
        # Setup logger
        # https://docs.python.org/3.6/library/logging.html#
        # logging levels: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler('IBAlgoStrategy.log')
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.info('Starting log at {}'.format(datetime.datetime.now()))

        # Create IB connection
        self.ib = self.connect()

        # Create empty list of instruments, bars, and indicators to track
        self.instruments = []
        self.bars = []
        self.bars_minutes = [] # keep list of bar minutes as ints
        self.indicators = []

        # Create empty dictionary of DataFrames for instrument bars
        self.dfs = {}

        # Create empty dictionary for trailing stop's enabled (for instruments)
        self.trailing_stop_enabled = {}
        # Create empty dictionary for best profit achived (for instruments)
        self.trade_profit_high = {}


###############################################################################
    def run(self):
        """Run logic for live trading"""
        # Get today's trading hours
        self.get_trading_hours()

        # Wait until the markets open
        open_hr = int(self.exchange_open[:2])
        open_min = int(self.exchange_open[-2:])
        self.log('Waiting until the market opens ({}:{} ET)...'.format(
            open_hr, open_min))
        while True:
            now = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
            if ((now.minute >= open_min) and (now.hour >= open_hr)) or \
               (now.hour >= open_hr+1):
                break

            # Sleep 5 seconds
            self.ib.sleep(5)

        # Exchange is now open
        self.log()
        self.log('The market is now open {}'.format(now))
        self.start_time = time.time()

        # Set time reference 9:30AM ET for today / 13:30 UTC time
        # Used for determining when new bars are available
        self.time_ref = calendar.timegm(time.strptime(
                now.strftime('%Y-%m-%d') + ' 13:30:00', '%Y-%m-%d %H:%M:%S'))

        # Run loop during exhange hours
        while True:
            # Get the time delta from the time reference
            time_since_open = time.time() - self.time_ref

            # Check for new intraday bars
            if len(self.bars_minutes)>0:
                for minutes in self.bars_minutes:
                    if 60*minutes-(time_since_open%(60*minutes)) <= 5:
                        # Time to update 'minutes' bar for all instruments
                        if minutes == 1:
                            bar = '1 min'
                        elif minutes < 60:
                            bar = str(minutes) + ' mins'
                        elif minutes == 60:
                            bar = '1 hour'
                        else:
                            hours =  minutes/60
                            bar = str(hours) + ' hours'
                        # Loop through all instruments and update bar dfs
                        for instrument in self.instruments:
                            # Get current df for instrument/bar
                            df = self.dfs[instrument][bar]
                            # Update instrument/bar df
                            df = self.update_bar(df, instrument, bar)
                            self.dfs[instrument][bar] = df
                            self.log("Updated {}'s {} df".format(
                                    instrument.symbol, bar))
                        # Process signals (new bar)
                        self.on_data()

            # Perform other checks once per minute (60s)
            if True:                                                            #5-(time_since_open%5) <= 5: #CONFIRM THIS IS CORRECT
                # Loop through instruments                                      #if 60-(time_since_open%60) <= 5: (check every minute, 5s margin of error)
                for instrument in self.instruments:                             #Replaced with Constant check.
                    # Get current qty
                    qty = self.get_quantity(instrument)
                    # Check for trailing exit signal
                    if qty != 0:
                        if self.trailing_exit_signal(instrument, qty):
                            # Go flat
                            self.go_flat(instrument)

            # Get current ET time
            now = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))

            # Get number of minutes until the market close
            close_str = now.strftime('%Y-%m-%d') + "T" \
                        + self.exchange_close[:2] \
                        + ":" + self.exchange_close[-2:] + ":00-04:00"
            close_time = datetime.datetime.strptime(
                ''.join(close_str.rsplit(':', 1)), '%Y-%m-%dT%H:%M:%S%z')
            min_to_close = int((close_time-now).seconds/60)

            #Exit for EOD!                                                      #ADDED EOD EXIT
            if min_to_close <= 30:
                if qty != 0:
                    self.go_flat(instrument)


            # Check for exchange closing time
            if min_to_close <= 0:
                log('The market is now closed: {}'.format(now))
                break

            # Sleep
            self.ib.sleep(5)

        self.log('Algo no longer running for the day.')


###############################################################################
    def on_data(self):
        """Process signals for new bar"""
        #self.log('on_data()')
        # Loop through instruments
        for instrument in self.instruments:
            # Check for any open orders
            #open_orders = self.get_open_orders(instrument)

            # Get current qty
            qty = self.get_quantity(instrument)
            self.log('Current qty for {}: {}'.format(instrument.symbol, qty))

            # Process current long position
            if qty > 0:
                # Check for short entry signal
                if self.short_entry_signal(instrument):
                    # Reverse position and go short
                    self.go_short(instrument)

                # Check for long exit signal
                elif self.long_exit_signal(instrument):
                    # Go flat
                    self.go_flat(instrument)

            # Process current short position
            elif qty < 0:
                # Check for long entry signal
                if self.long_entry_signal(instrument):
                    # Reverse position and go long
                    self.go_long(instrument)

                # Check for short exit signal
                elif self.short_exit_signal(instrument):
                    # Go flat
                    self.go_flat(instrument)

            # Check for entry signal
            else:
                # Check for long entry signal
                if self.long_entry_signal(instrument):
                    # Go long
                    self.go_long(instrument)
                # Check for short entry signal
                elif self.short_entry_signal(instrument):
                    # Go short
                    self.go_short(instrument)



###############################################################################  --Modified for BBs
    def long_entry_signal(self, instrument):
        """
        Check for LONG entry signal for the instrument.
        Returns True or False.
        """
        # Always use the first bar for this signal
        bar = self.bars[0]

        # Get intrument/bar df
        df = self.dfs[instrument][bar]

        # Get last close, PSAR, and RSI
        price = df['close'].iloc[-1]
        RSI   = df['RSI'].iloc[-1]
        HH = df['HH'].iloc[-1]

        # Entry logic
        if condition:
            self.log('{} LONG ENTRY SIGNAL: price={}, HH={}'.format(
                    instrument.symbol, price, HH))
            return True
        return False


############################################################################### --Modified for BBs
    def long_exit_signal(self, instrument):
        """
        Check for LONG exit signal for the instrument.
        Returns True or False.
        """
        # Always use the first bar for this signal
        bar = self.bars[0]

        # Get intrument/bar df
        df = self.dfs[instrument][bar]

        # Get last close, PSAR, and RSI
        price = df['close'].iloc[-1]
        PSAR  = df['PSAR'].iloc[-1]



        # Exit logic
        if condition:
            self.log('{} LONG BB EXIT SIGNAL: price={}, PSAR={}'.format(
                    instrument.symbol, price, PSAR))
            return True
        return False


############################################################################### -- Modified for BBS
    def short_entry_signal(self, instrument):
        """
        Check for SHORT entry signal for the instrument.
        Returns True or False.
        """
        # Always use the first bar for this signal
        bar = self.bars[0]

        # Get intrument/bar df
        df = self.dfs[instrument][bar]

        # Get last close, PSAR, and RSI
        price = df['close'].iloc[-1]
        RSI   = df['RSI'].iloc[-1]





        # Entry logic
        if condition:
            self.log('{} SHORT ENTRY SIGNAL: price={},LL={}'.format(
                    instrument.symbol, price, LL))
            return True
        return False


############################################################################### -- Modified for BBS
    def short_exit_signal(self, instrument):
        """
        Check for SHORT exit signal for the instrument.
        Returns True or False.
        """
        # Always use the first bar for this signal
        bar = self.bars[0]

        # Get intrument/bar df
        df = self.dfs[instrument][bar]

        # Get last close, PSAR, and RSI
        price = df['close'].iloc[-1]
        PSAR  = df['PSAR'].iloc[-1]



        # Exit logic
        if condition:
            self.log('{} SHORT BB EXIT SIGNAL: price={}, PSAR={}'.format(
                    instrument.symbol, price, PSAR))
            return True
        return False


###############################################################################
    def trailing_exit_signal(self, instrument, qty):
        """
        Check for trailing exit signal for the instrument.
        Also checks for the initial stop loss exit signal for the instrument.
        Returns True or False.
        """
        self.log('Checking for stop loss or trailing stop exit for {}'.format(
                instrument.symbol))

        # Get cost basis for instrument
        cost_basis = self.get_cost_basis(instrument)
        dollar_cost = cost_basis*qty

        # Get latest prices for instrument
        bid, ask, mid = self.get_price(instrument)
        # Verify valid prices
        if mid is None:
            return False

        # Calculate current dollar profit (Per Position basis, not contract)
        if qty > 0:
            # Long position
            profit = (mid-cost_basis)*qty
        elif qty < 0:
            # Short position
            profit = (cost_basis-mid)*qty

        # Check if the stop loss is triggered
        if profit <= -abs(SL):
            self.log('Stop loss exit triggered for {}, profit={}'.format(
                    instrument.symbol, profit))
            return True

        # Check if trailing stop should be enabled (PT reached)
        if not self.trailing_stop_enabled[instrument]:
            if profit >= PT:
                # Set trailing stop to be enabled and save high profit
                self.trailing_stop_enabled[instrument] = True
                self.trade_profit_high[instrument] = 0
                PNL[instrument] += profit                                       #Added PNL Increment
                self.log('Trailing stop for {} enabled, profit={}'.format(
                        instrument.symbol, profit))
            # Always return False for exit signal if not previously turned on
            return False

        # Trailing stop previously enabled, so check for exit signal
        # Check for new profit high
        if profit > self.trade_profit_high[instrument]:
            # Update trade profit and return False for signal
            self.trade_profit_high[instrument] = profit
            return False

        # Check for exit signal
        exit_trigger = (1.0-PCT)*self.trade_profit_high[instrument]
        if profit < exit_trigger:
            PNL[instrument] += profit                                           #Added PNL Increment
            self.log("Trailing exit triggered for {}: profit={}, "
                     "high profit={}".format(
                             instrument.symbol, profit,
                             self.trade_profit_high[instrument]))
            return True
        return False


###############################################################################
    def get_price(self, instrument):
        """Get the current bid, ask, and mid price for an instrument"""
        # Create IB object to request price info
        ticker = self.ib.reqMktData(instrument, "", False, False)
        # Wait 2 seconds
        self.ib.sleep(2.0)
        # Loop until getting bid and ask or 20s max
        bid = None
        ask = None
        for i in range(100):
            self.ib.sleep(0.2)
            if ticker.bid is not None and ticker.ask is not None:
                bid = float(ticker.bid)
                ask = float(ticker.ask)
                break

        # Check for valid bid and ask
        try:
            mid = round((bid+ask)/2,2)
        except:
            self.log('Error getting current bid/ask prices for {}'.format(
                    instrument))
            return None, None, None

        #self.log('{} current bid={}, ask={}, mid={}'.format(
        #        instrument.symbol, bid, ask, mid))
        return bid, ask, mid


###############################################################################
    def go_long(self, instrument):
        """Go LONG instrument"""
        # Get current qty
        qty = self.get_quantity(instrument)
# Get desired qty - no function for this currently
        #desired_qty = QTY
        #desired_qty = self.get_FR_qty(instrument)

        desired_qty = self.get_fxd_qty(instrument)
        # Get order quantity
        order_qty = desired_qty-qty
        # Place market order to go long
        self.market_order(instrument, 'BUY', abs(order_qty))
        # Set trailing stop enabled to False for instrument
        self.trailing_stop_enabled[instrument] = False
        # Set trade profit high to date to be zero
        self.trade_profit_high[instrument] = 0


###############################################################################
    def go_short(self, instrument):
        """Go SHORT instrument"""
        # Get current qty
        qty = self.get_quantity(instrument)
# Get desired qty - no function for this currently
        #desired_qty = -QTY
        #desired_qty = -(self.get_FR_qty(instrument))
        desired_qty = -(self.get_fxd_qty(instrument))
        # Get order quantity
        order_qty = desired_qty-qty
        # Place market order to go short
        self.market_order(instrument, 'SELL', abs(order_qty))
        # Set trailing stop enabled to False for instrument
        self.trailing_stop_enabled[instrument] = False
        # Set trade profit high to date to be zero
        self.trade_profit_high[instrument] = 0


###############################################################################
    def go_flat(self, instrument):
        """Go FLAT instrument"""
        # Get current qty
        qty = self.get_quantity(instrument)
        # Place market order to go flat
        if qty > 0:
            action = 'SELL'
        elif qty < 0:
            action = 'BUY'
        else:
            self.log('{} already FLAT.'.format(instrument.symbol))
            return
        # Place market order to go flat
        self.market_order(instrument, action, abs(qty))


###############################################################################
    def get_open_orders(self, instrument):
        """
        Checks for any open orders for instrument.
        Returns True or False.
        """
        # Verify open orders match open trades
        for i in range(10):
            open_trades = list(self.ib.openTrades())
            trade_ids = set([t.order.orderId for t in open_trades])
            open_orders = list(self.ib.reqOpenOrders())
            order_ids = set([o.orderId for o in open_orders])
            missing = order_ids.difference(trade_ids)
            if len(missing) == 0 and len(open_trades) > 0:
                break

        # Return True if any open trade is for instrument, otherwise False
        for trade in open_trades:
            if instrument.symbol == trade.contract.localSymbol:
                return True
        return False


###############################################################################
    def get_quantity(self, instrument):
        """Returns the current quantity held for instrument"""
        # Get instrument type
        if str(type(instrument))[-7:-2] == 'Stock':
            instrument_type = 'Stock'
        elif str(type(instrument))[-7:-2] == 'Option':
            instrument_type = 'Option'
        elif str(type(instrument))[-7:-2] == 'Future':
            instrument_type = 'Future'
        else:
            raise ValueError('Invalid instrument type ({}) for '
                             'current_quantity'.format(type(instrument)))

        # Loop through all current positions
        for position in self.ib.positions():
            # Verify position is for instrument
            contract = position.contract
            if instrument_type == 'Stock':
                try:
                    if contract.secType == 'STK' and \
                        contract.localSymbol == instrument.symbol:
                        return int(position.position)
                except:
                    continue
            elif instrument_type == 'Option':
                pass
            elif instrument_type == 'Future':
                pass

        return 0


###############################################################################
    def get_cost_basis(self, instrument):
        """Returns the current cost basis for an instrument's position"""
        # Get instrument type
        if str(type(instrument))[-7:-2] == 'Stock':
            instrument_type = 'Stock'
        elif str(type(instrument))[-7:-2] == 'Option':
            instrument_type = 'Option'
        elif str(type(instrument))[-7:-2] == 'Future':
            instrument_type = 'Future'
        else:
            raise ValueError('Invalid instrument type ({}) for '
                             'current_quantity'.format(type(instrument)))

        # Loop through all current positions
        for position in self.ib.positions():
            # Verify position is for instrument
            contract = position.contract
            if instrument_type == 'Stock':
                try:
                    if contract.secType == 'STK' and \
                        contract.localSymbol == instrument.symbol:
                        return float(position.avgCost)
                except:
                    continue
            elif instrument_type == 'Option':
                pass
            elif instrument_type == 'Future':
                pass

        return 0


###############################################################################
    def market_order(self, instrument, action, qty):
        """Place market order for instrument"""
        # Verify action
        if action != 'BUY' and action != 'SELL':
            raise ValueError("Invalid action () for market order. Must be "
                             "'BUY' or 'SELL'.".format(action))

        market_order = MarketOrder(
                action=action,
                totalQuantity=float(qty)
                )
        self.log('{}ING {} units of {} at MARKET'.format(
                action, qty, instrument.symbol))
        self.ib.placeOrder(instrument, market_order)


###############################################################################
    def limit_order(self, instrument, action, qty, limit_price):
        """Place limit order for instrument"""
        # Verify action
        if action != 'BUY' and action != 'SELL':
            raise ValueError("Invalid action () for market order. Must be "
                             "'BUY' or 'SELL'.".format(action))

        limit_order = LimitOrder(
                action=action,
                totalQuantity=float(qty),
                lmtPrice=float(limit_price)
                )
        self.log('{}ING {} units of {} at {} LIMIT'.format(
                action, qty, instrument.symbol, limit_price))
        self.ib.placeOrder(instrument, limit_order)


###############################################################################
    def connect(self):
        """Connect to Interactive Brokers TWS"""
        self.log('Connecting to Interactive Brokers TWS...')
        try:
            if USING_NOTEBOOK:
                util.startLoop()
            ib = IB()
            ib.connect('127.0.0.1', 7497, clientId=1)
            self.log('Connected')
            self.log()
            return ib
        except:
            self.log('Error in connecting to TWS!! Exiting...')
            self.log(sys.exc_info()[0])
            exit(-1)

###############################################################################
    def log(self, msg=""):
        """Add log to output file"""
        self.logger.info(msg)
        if not USING_NOTEBOOK:
            print(msg)


###############################################################################
    def get_trading_hours(self):
        """Get today's trading hours for algo's instruments"""
        open_time = None
        close_time = None

        # Get todays date in YYYYMMDD format
        today = datetime.datetime.today().strftime('%Y%m%d')

        # Loop through instruments
        for instrument in self.instruments:
            # Get contract details
            contract_details = self.ib.reqContractDetails(instrument)[0]
            # Get regular trading hours for today
            trading_hours_list = contract_details.liquidHours.split(';')
            for item in trading_hours_list:
                if item[:8] == today:
                    # Update open time
                    if open_time is None:
                        open_time = item.split('-')[0].split(':')[1]
                    else:
                        if item.split('-')[0].split(':')[1] < open_time:
                            open_time = item.split('-')[0].split(':')[1]
                    # Update close time
                    if close_time is None:
                        close_time = item.split('-')[1].split(':')[1]
                    else:
                        if item.split('-')[1].split(':')[1] > close_time:
                            close_time = item.split('-')[1].split(':')[1]
                    break

        # Save earliest start time
        self.exchange_open = open_time
        # Save latest end time
        self.exchange_close = close_time
        self.log()
        self.log("Today's exchange hours are {}-{}".format(
                self.exchange_open, self.exchange_close))


###############################################################################
    def add_instrument(self, instrument_type, ticker, last_trade_date=None,
                       strike=None, option_type=None, routing='SMART',
                       currency='USD', primary_exchange='NASDAQ',
                       future_exchange='GLOBEX'):
        """
        Add instrument to trade to algo.
        https://github.com/erdewit/ib_insync/blob/master/ib_insync/contract.py
        """
        if instrument_type == 'Stock':
            instrument = Stock(ticker, routing, currency,
                               primaryExchange=primary_exchange)
        elif instrument_type == 'Option':
            # Verify option type
            valid_option_types = ['C', 'P', 'CALL', 'PUT']
            if option_type not in valid_option_types:
                raise ValueError(
                        'Invalid option_type: {}. Must be in {}'.format(
                                option_type, valid_option_types))
            instrument = Option(ticker, last_trade_date, float(strike),
                                option_type, routing)
        elif instrument_type == 'Future':
            instrument = Future(ticker, last_trade_date, future_exchange)
        else:
            raise ValueError(
                    "Invalid instrument type: {}".format(instrument_type))

        # Append instrument to algo list
        self.ib.qualifyContracts(instrument)
        self.instruments.append(instrument)

        # Create dictionary for instrument bars
        self.dfs[instrument] = {}


###############################################################################
    def add_bar(self, bar):
        """Add bar to algo"""
        # https://interactivebrokers.github.io/tws-api/historical_bars.html
        valid_bars = ['1 min','2 mins','3 mins','10 mins','20 mins','30 mins',
                      '1 hour','2 hours','3 hours','4 hours','8 hours',
                      '1 day','1 week','1 month']
        # Verify bar size
        if bar not in valid_bars:
            raise ValueError('Invalid bar: {}. Must be in {}'.format
                             (bar, valid_bars))
        # Append bar to algo list
        self.bars.append(bar)

        # Get bar minutes (when applicable)
        if bar[-3:] == 'min' or bar[-4:] == 'mins':
            # min bar
            bar_minutes = int(bar.split(' ')[0])
            self.bars_minutes.append(bar_minutes)
        elif bar[-4:] == 'hour' or bar[-5:] == 'hours':
            # hourly bar
            bar_minutes = int(60*bar.split(' ')[0])
            self.bars_minutes.append(bar_minutes)

        # Initialize dfs for all instruments
        for instrument in self.instruments:
            # Get ohlc pandas DataFrame
            df = self.get_historical_data(instrument, bar)
            # Add indicators to df and save to algo
            self.dfs[instrument][bar] = self.add_indicators(df)


###############################################################################
    def get_historical_data(self, instrument, bar, end_date="", use_RTH=True):
        """
        Get historical bars for instrument.
        https://interactivebrokers.github.io/tws-api/historical_bars.html
        """
        # Get max duration for a given bar size
        if bar in ['1 day','1 week','1 month']:
            duration = '1 Y' # one year
        elif bar in ['30 mins','1 hour','2 hours','3 hours','4 hours','8 hours']:
            duration = '1 M' # one month
        elif bar in ['3 mins','10 mins','20 mins']:
            duration = '1 W' # one week
        elif bar == '2 mins':
            duration = '2 D' # two days
        elif bar == '1 min':
            duration = '1 D' # one day
        else:
            raise ValueError(
                    'Invalid bar: {} for get_historical_data()'.format(bar))

        # Get historical bars
        bars = self.ib.reqHistoricalData(
            instrument,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=bar,
            whatToShow='TRADES',
            useRTH=use_RTH) # use regular trading hours

        # Convert bars to a pandas dataframe
        hist = util.df(bars)

        # Make the 'date' column the index
        hist.set_index('date', inplace=True)

        # Remove the 'volume', 'barCount', and 'average' columns
        hist.drop(columns=['volume', 'barCount', 'average'], inplace=True)

        # Add TWS time zone to datetimes
        hist = hist.tz_localize(TWS_TIMEZONE)

        return hist


###############################################################################
    def update_bar(self, df, instrument, bar, end_date="", use_RTH=True):
        """
        Update historical bars for instrument df.
        """
        # Set duration for 1 day
        duration = '1 D' # one day

        # Get historical bars
        bars = self.ib.reqHistoricalData(
            instrument,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=bar,
            whatToShow='TRADES',
            useRTH=use_RTH) # use regular trading hours

        # Convert bars to a pandas dataframe
        hist = util.df(bars)

        # Make the 'date' column the index
        hist.set_index('date', inplace=True)

        # Remove the 'volume', 'barCount', and 'average' columns
        hist.drop(columns=['volume', 'barCount', 'average'], inplace=True)

        # Add TWS time zone to datetimes
        hist = hist.tz_localize(TWS_TIMEZONE)

        # Check if hist last date != df last date
        if hist.index[-1] != df.index[-1]:
            # Append new bar to df
            try:
                df = df.append(hist, sort=True)
            except:
                df = df.append(hist)
            # Add indicators to df
            df = self.add_indicators(df)

        return df


###############################################################################
    def add_indicators(self, df):
        """Add technical indicators to pandas DataFrame"""
        # Check for RSI in indicators
        if 'RSI' in self.indicators:
            # Add RSI to df
            df = self.get_RSI(df)
        # Check for Parabolic SAR in indicators
        if 'SAR' in self.indicators:
            # Add SAR to df
            df = self.get_SAR(df)
        # Check for ATR in indicators
        if 'ATR' in self.indicators:
            df = self.get_ATR(df)
        if 'HL' in self.indicators:
            df = self.get_HL(df)

        return df


###############################################################################
    def add_RSI(self, length, alpha):
        """Add the RSI indicator to the list of indicators."""
        # Verify correct alpha input
        valid_alpha_list = ['Wilders', 'Standard']
        if alpha not in valid_alpha_list:
            raise ValueError('Invalid RSI alpha input ({}). Must be {}'.format(
                    alpha, alpha_list))
        # Save RSI input parameters to algo
        self.RSI_length = length
        self.RSI_alpha  = alpha
        # Add RSI to list of indicators
        self.indicators.append('RSI')


###############################################################################
    def get_RSI(self, df):
        """Add the RSI calculations to pandas DataFrame"""
        # Using Wilder's EMA: alpha=1/n instead of standard alpha = 2/(n+1)
        df['delta'] = df['close'].diff()
        df['gain'], df['loss'] = df['delta'].copy(), df['delta'].copy()
        df['gain'][df['gain']<0] = 0
        df['loss'][df['loss']>0] = 0

        # Get alpha for RSI EMA
        if self.RSI_alpha == 'Wilders': # alpha=1/n
            alpha = float(1.0/self.RSI_length)
        elif self.RSI_alpha == 'Standard':
            alpha = float(2.0/(self.RSI_length+1))

        # Calculate the avg gain and loss
        df['avg_gain'] = df['gain'].ewm(alpha=alpha).mean()
        df['avg_loss'] = df['loss'].ewm(alpha=alpha).mean()

        # Calculate RS and RSI
        df['RS'] = abs(df['avg_gain']/df['avg_loss'])
        df['RSI'] = 100.0-(100.0/(1+df['RS']))
        return df

############################################################################### ---Addit

    def add_HL(self,length):
        '''Adds HH LL to list of indicators to init'''
        if length < 2 or length > 10:
            raise ValueError('Invalid Length {}; Length must be between 2 and 10'.format(length))
        self.HL_Len = length
        self.indicators.append('HL')

############################################################################### ---Addit

    def get_HL(self,df):
        '''Adds HH LL (both) to pd DF'''
        c = df['close']
        #High
        for i in range(self.HL_Len - 1):
            if c.iloc[-i] < c.iloc[-i -1]:
                df['HH'] = False
            df['HH'] = True
        for i in range(self.HL_Len - 1):
            if c.iloc[-i] > c.iloc[-i -1]:
                df['LL'] = False
            df['LL'] = True
        return df




###############################################################################
    def add_SAR(self, af, af_max):
        """Add the Parabolic SAR indicator to the list of indicators."""
        # Save SAR input parameters to algo
        self.SAR_af = af
        self.SAR_af_max = af_max
        # Add SAR to list of indicators
        self.indicators.append('SAR')


###############################################################################
    def get_SAR(self, df):
        """
        Add the Parabolic SAR indicator to pandas DataFrame
        ref: https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
        ref: https://stackoverflow.com/questions/54918485/parabolic-sar-in-python-psar-keep-growing-instead-of-reversing
        """
        # Reset index
        df.reset_index(inplace=True)

        # Initialize df columns first values
        df.loc[0,'AF'] = self.SAR_af
        df.loc[0,'PSAR'] = df.loc[0,'low']
        df.loc[0,'EP'] = df.loc[0,'high']
        df.loc[0,'PSAR_dir'] = "bull"

        # Loop through rows of df
        for a in range(1, len(df)):
            # Previous uptrend
            if df.loc[a-1,'PSAR_dir'] == 'bull':
                # Calculate the current PSAR (rising)
                # = previous PSAR + previous AF(previous EP-previous PSAR)
                df.loc[a,'PSAR'] = df.loc[a-1,'PSAR'] + \
                    df.loc[a-1,'AF']*(df.loc[a-1,'EP']-df.loc[a-1,'PSAR'])

                # Check for change in trend
                if df.loc[a,'low'] < df.loc[a-1,'PSAR']:
                    # now downtrend
                    df.loc[a,'PSAR_dir'] = "bear"
                    df.loc[a,'PSAR'] = df.loc[a-1,'EP']
                    df.loc[a,'EP'] = df.loc[a-1,'low']
                    df.loc[a,'AF'] = self.SAR_af

                # Update uptrend SAR
                else:
                    df.loc[a,'PSAR_dir'] = "bull"
                    # Check for extreme point (high)
                    if df.loc[a,'high'] > df.loc[a-1,'EP']:
                        # Update extreme point
                        df.loc[a,'EP'] = df.loc[a,'high']
                        # Increase AF if below max
                        if df.loc[a-1,'AF'] <= self.SAR_af_max-self.SAR_af:
                            df.loc[a,'AF'] = df.loc[a-1,'AF'] + self.SAR_af
                        else:
                            df.loc[a,'AF'] = df.loc[a-1,'AF']
                    # No new extreme point
                    elif df.loc[a,'high'] <= df.loc[a-1, 'EP']:
                        df.loc[a,'AF'] = df.loc[a-1,'AF']
                        df.loc[a,'EP'] = df.loc[a-1,'EP']

            # Previous downtrend
            elif df.loc[a-1,'PSAR_dir'] == 'bear':
                # Calculate the current PSAR (falling)
                # = previous PSAR - previous AF(previous PSAR-previous EP)
                df.loc[a,'PSAR'] = df.loc[a-1,'PSAR'] - \
                    (df.loc[a-1,'AF']*(df.loc[a-1,'PSAR']-df.loc[a-1,'EP']))

                # Check for change in trend
                if df.loc[a,'high'] > df.loc[a-1,'PSAR']:
                    # now uptrend
                    df.loc[a,'PSAR_dir'] = "bull"
                    df.loc[a,'PSAR'] = df.loc[a-1,'EP']
                    df.loc[a,'EP'] = df.loc[a-1,'high']
                    df.loc[a,'AF'] = self.SAR_af

                # Update downtrend SAR
                else:
                    df.loc[a,'PSAR_dir'] = "bear"
                    # Check for new extreme point (low)
                    if df.loc[a,'low'] < df.loc[a-1, 'EP']:
                        # Update extreme point
                        df.loc[a,'EP'] = df.loc[a, 'low']
                        # Increase AF if below max
                        if df.loc[a-1,'AF'] <= self.SAR_af_max-self.SAR_af:
                            df.loc[a,'AF'] = df.loc[a-1,'AF'] + self.SAR_af
                        else:
                            df.loc[a,'AF'] = df.loc[a-1,'AF']
                    # No new extreme point
                    elif df.loc[a,'low'] >= df.loc[a-1,'EP']:
                        df.loc[a,'AF'] = df.loc[a-1,'AF']
                        df.loc[a,'EP'] = df.loc[a-1,'EP']

        # Reset date to be the index
        df.set_index('date', inplace=True)

        return df


###############################################################################
    def add_ATR(self, length):
        """Add the ATR indicator to the list of indicators."""
        # Save ATR input parameters to algo
        self.ATR_length = length
        # Add ATR to list of indicators
        self.indicators.append('ATR')


###############################################################################
    def get_ATR(self, df):
        """Add the ATR calculations to pandas DataFrame"""
        df['tr1'] = df['high'] - df['low'] # H-L
        df['tr2'] = abs(df['high'] - df['close'].shift()) # abs(H-prev_close)
        df['tr3'] = abs(df['low'] - df['close'].shift()) # abs(L-prev_close)
        df['tr'] = df[['tr1', 'tr2', 'tr3']].apply(max,axis=1) #axis=1 for row max, axis=0 for column max
        df['atr'] = df['tr'].rolling(self.ATR_length).mean()
        return df


############################################################################### #ADDED
    '''DYNAMIC VERSION OF FIXED QTY IN USD -- CALCULATES EACH TIME'''

    def get_dyn_fxd(self,instrument):
        '''Dynamic version of fxd qty'''
        # Always use the first bar for this signal
        bar = self.bars[0]
        # Get intrument/bar df
        df = self.dfs[instrument][bar]

        c = df['close'].iloc[-1]
        QTY_DICT[instrument] = round(POS_SIZE_USD / c,0)
        return int(QTY_DICT[instrument])



    def create_fxd_dict(self,instrument_list):
        for sec in instrument_list:
            QTY_DICT[sec] = 0
        return QTY_DICT



############################################################################### #ADDED

    def get_fxd_qty(self,instrument):
        '''Get Fixed Qty Per instrument -- BASIC'''
        #qty = POS_SIZE_USD / df[instrument]['close']
        qty = round(POS_SIZE_USD / CLOSE_DICT[instrument],0)
        return qty

############################################################################### #ADDED

    def create_PNL(self,instrument_list):
        '''Create Dict of PNL[sec] = sec_pnl (maybe later DF) '''
        for sec in instrument_list:
            PNL[sec] = 0
            #pnl[sec] = [] #To track all wins/losses, + pnl[sec].append(profit)
        return PNL

############################################################################### #ADDED

    def get_FR_qty(self,instrument):
        '''Accesses PNL Dict and Returns desired Qty'''
        sec_pnl = PNL[instrument]
        q = .5 * (np.sqrt(1 + 8*(sec_pnl/DELTA)) + 1)
        return q


###############################################################################
if __name__ == '__main__':
    # Create algo object
    algo = IBAlgoStrategy()

    # Add instruments to trade
    algo.add_instrument('Stock', 'SPY')
    #algo.add_instrument('Option', 'SPY', '20190920', 300, 'C')
    #algo.add_instrument('Future', 'ES', '20190920')
    PNL = algo.create_PNL(['SPY'])

    algo.create_fxd_dict(['SPY'])


    # Add RSI to indicators
    algo.add_RSI(length=14, alpha='Wilders')

    # Add ATR to indicators
    algo.add_ATR(length=14)



    # Add intraday bars
    #valid_bars = ['1 min','2 mins','3 mins','10 mins','20 mins','30 mins','1 hour','2 hours','3 hours','4 hours','8 hours','1 day','1 week','1 month']
    #algo.add_bar('1 hour')
    algo.add_bar('1 hour')

    # Run algo for the day
    algo.run()
