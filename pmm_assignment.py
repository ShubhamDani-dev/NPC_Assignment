import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class PMM(ScriptStrategyBase):
    """
    Enhanced Pure Market Making Strategy
    
    Combines three key components:
    1. Volatility-based spread adjustment using NATR
    2. Trend-based price shifting using RSI
    3. Inventory management with target ratio
    4. Risk management framework with position sizing and stop-loss
    """
    
    # Basic strategy parameters
    bid_spread = 0.0001
    ask_spread = 0.0001
    order_refresh_time = 30
    order_amount = 0.01
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')
    
    # Candles parameters
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000
    
    # Volatility spread parameters
    bid_spread_scalar = 120
    ask_spread_scalar = 60
    
    # Trend shift parameters
    max_shift_spread = 0.000001
    trend_scalar = -1
    price_multiplier = 0
    
    # Inventory parameters
    target_ratio = 0.5
    current_ratio = 0.5
    inventory_scalar = 1
    inventory_multiplier = 0
    
    # Risk parameters
    max_position_size = 0.05
    min_position_size = 0.005
    stop_loss_pct = -0.05
    emergency_exit_pct = -0.07
    max_inventory_value = 1150000
    
    # Initialize candles
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector=candle_exchange,
            trading_pair=trading_pair,
            interval=candles_interval,
            max_records=max_records
        )
    )
    
    # Define markets required for the strategy
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """Initialize the strategy"""
        super().__init__(connectors)
        self.candles.start()
        self.orig_price = Decimal("0")
        self.reference_price = Decimal("0")
        self.entry_price = None
        self.last_checked_position = 0
        self.log_with_clock(logging.INFO, "Enhanced PMM strategy initialized")
    
    def on_stop(self):
        """Stop candles when strategy stops"""
        self.candles.stop()

    def on_start(self):
        self.max_inventory_value = Decimal("150000")
    
    def on_tick(self):
        """Main execution logic on each tick"""
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            
            # Execute risk check before placing new orders
            if self.check_stop_loss():
                self.log_with_clock(logging.WARNING, f"Stop loss triggered! Exiting position.")
                return

            self.update_multipliers()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
    
    def get_candles_with_features(self):
        """Add technical indicators to candles dataframe"""
        candles_df = self.candles.candles_df
        
        # Add NATR for volatility calculation
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
        candles_df['bid_spread_bps'] = candles_df[f"NATR_{self.candles_length}"] * self.bid_spread_scalar * 10000
        candles_df['ask_spread_bps'] = candles_df[f"NATR_{self.candles_length}"] * self.ask_spread_scalar * 10000
        
        # Add RSI for trend detection
        candles_df.ta.rsi(length=self.candles_length, append=True)
        
        return candles_df
    
    def update_multipliers(self):
        """Update all multipliers for spreads and price shifts"""
        candles_df = self.get_candles_with_features()
        
        # 1. Volatility-based spread adjustment
        current_natr = candles_df[f"NATR_{self.candles_length}"].iloc[-1]
        self.bid_spread = current_natr * self.bid_spread_scalar
        self.ask_spread = current_natr * self.ask_spread_scalar
        
        # 2. Trend-based price shift using RSI
        rsi = candles_df[f"RSI_{self.candles_length}"].iloc[-1]
        self.price_multiplier = (rsi - 50) / 50 * self.max_shift_spread * self.trend_scalar
        
        # 3. Inventory-based price shift
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.connectors[self.exchange].get_price_by_type(
            self.trading_pair, self.price_source
        )
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        
        # Calculate current inventory ratio
        if base_bal_in_quote + quote_bal > 0:
            self.current_ratio = float(base_bal_in_quote / (base_bal_in_quote + quote_bal))
        else:
            self.current_ratio = 0.5
            
        # Calculate inventory delta and multiplier
        delta = (self.target_ratio - self.current_ratio) / self.target_ratio
        inventory_delta = max(-1, min(1, delta))
        self.inventory_multiplier = inventory_delta * self.max_shift_spread * self.inventory_scalar
        
        # 4. Volatility-based position sizing
        self.order_amount = self.calculate_position_size(current_natr)
        
        # 5. Reference price calculation with all adjustments
        self.orig_price = self.connectors[self.exchange].get_price_by_type(
            self.trading_pair, self.price_source
        )
        self.reference_price = self.orig_price * Decimal(str(1 + self.price_multiplier)) * Decimal(str(1 + self.inventory_multiplier))
        
        # Track entry price for stop-loss
        if self.entry_price is None:
            self.entry_price = self.orig_price
    
    def calculate_position_size(self, volatility):
        """Adjust position size based on volatility"""
        # Calculate dynamic position size (smaller in high volatility)
        if volatility <= 0:
            return self.max_position_size
            
        # Inverse relationship between volatility and position size
        position_size = self.max_position_size / (volatility * 2)
        
        # Apply constraints
        position_size = min(self.max_position_size, max(self.min_position_size, position_size))
        
        return position_size
    
    def check_stop_loss(self):
        """Check if stop-loss should be triggered"""
        # Skip frequent checks
        if self.current_timestamp - self.last_checked_position < 30:
            return False
            
        self.last_checked_position = self.current_timestamp
        
        # If we don't have an entry price yet, can't check stop loss
        if self.entry_price is None:
            return False
            
        current_price = self.connectors[self.exchange].get_price_by_type(
            self.trading_pair, self.price_source
        )
        
        # Check if price has dropped below stop loss threshold
        drawdown = float(current_price / self.entry_price - 1)
        
        # Emergency exit - liquidate immediately
        if drawdown <= self.emergency_exit_pct:
            self.log_with_clock(
                logging.WARNING, 
                f"Emergency stop-loss triggered! Drawdown: {drawdown:.2%}"
            )
            self.liquidate_position()
            return True
            
        # Regular stop loss - avoid placing new orders
        if drawdown <= self.stop_loss_pct:
            self.log_with_clock(
                logging.WARNING, 
                f"Stop-loss level reached. Drawdown: {drawdown:.2%}"
            )
            return True
            
        # Check max inventory value constraint
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_value = base_bal * current_price
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        total_value = float(base_value + quote_bal)
        
        if total_value > self.max_inventory_value:
            self.log_with_clock(
                logging.WARNING,
                f"Max inventory value reached! Current: ${total_value:.2f}, Max: ${self.max_inventory_value:.2f}"
            )
            return True
            
        return False
        
    def liquidate_position(self):
        """Emergency liquidation of position"""
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        if base_bal > 0:
            self.sell(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=base_bal,
                order_type=OrderType.MARKET,
                price=None
            )
    
    def create_proposal(self) -> List[OrderCandidate]:
        """Create buy and sell orders with adjusted prices"""
        # Ensure our orders are not more aggressive than the order book
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        # Calculate our order prices using the reference price and spreads
        buy_price = min(self.reference_price * Decimal(1 - self.bid_spread), best_bid)
        sell_price = max(self.reference_price * Decimal(1 + self.ask_spread), best_ask)
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(self.order_amount),
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(self.order_amount),
            price=sell_price
        )
        
        return [buy_order, sell_order]
    
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order sizes based on available budget"""
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=True
        )
        return proposal_adjusted
    
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders on the exchange"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
    
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place a single order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
    
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
    
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle filled order events"""
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        # Update entry price for stop loss calculation
        if self.entry_price is None:
            self.entry_price = event.price
    
    def format_status(self) -> str:
        """Format status for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # Display balances
        balance_df = self.get_balance_df()
        lines.extend(["", " Balances:"] + [" " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Display orders
        try:
            df = self.active_orders_df()
            lines.extend(["", " Orders:"] + [" " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", " No active maker orders."])
        
        # Display strategy metrics
        ref_price = self.reference_price
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Spreads:"])
        lines.extend([f" Bid Spread (bps): {self.bid_spread * 10000:.4f}"])
        lines.extend([f" Ask Spread (bps): {self.ask_spread * 10000:.4f}"])
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Price Adjustments:"])
        lines.extend([f" Max Shift (bps): {self.max_shift_spread * 10000:.4f}"])
        lines.extend([f" Trend Multiplier (bps): {self.price_multiplier * 10000:.4f}"])
        lines.extend([f" Inventory Target: {self.target_ratio:.2f} | Current: {self.current_ratio:.2f}"])
        lines.extend([f" Inventory Multiplier (bps): {self.inventory_multiplier * 10000:.4f}"])
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Risk Management:"])
        lines.extend([f" Position Size: {self.order_amount:.4f}"])
        if self.entry_price:
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            drawdown = float(current_price / self.entry_price - 1)
            lines.extend([f" Current Drawdown: {drawdown:.2%} | Stop Loss: {self.stop_loss_pct:.2%}"])
        
        return "\n".join(lines)
