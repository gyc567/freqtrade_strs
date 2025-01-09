from typing import Optional  # 用于类型注解，Optional表示可选类型

# 导入qtpylib库中的指标模块，用于技术指标计算
import freqtrade.vendor.qtpylib.indicators as qtpylib
# 导入numpy库，用于数值计算
import numpy as np
# 导入talib库中的抽象指标模块，用于技术分析
import talib.abstract as ta

# 从freqtrade.persistence模块中导入Trade类，用于交易记录管理
from freqtrade.persistence import Trade
# 从freqtrade.strategy.interface模块中导入IStrategy类，作为策略基类
from freqtrade.strategy.interface import IStrategy
# 从freqtrade.strategy模块中导入常用函数和参数类型
from freqtrade.strategy import (
    merge_informative_pair,  # 用于合并不同时间框架的数据
    DecimalParameter,        # 用于定义小数类型的参数
    stoploss_from_open,      # 用于计算相对于开仓价的止损
    RealParameter            # 用于定义实数类型的参数
)
# 从pandas库中导入DataFrame和Series类，用于数据处理
from pandas import DataFrame, Series
# 从datetime库中导入datetime类，用于时间处理
from datetime import datetime

# 定义一个函数，用于计算布林带指标
def bollinger_bands(stock_price, window_size, num_of_std):
    # 计算滚动平均值
    rolling_mean = stock_price.rolling(window=window_size).mean()
    # 计算滚动标准差
    rolling_std = stock_price.rolling(window=window_size).std()
    # 计算下轨
    lower_band = rolling_mean - (rolling_std * num_of_std)
    # 将NaN值替换为0并返回滚动平均值和下轨
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

# 定义一个函数，用于计算Heikin Ashi典型价格
def ha_typical_price(bars):
    # 计算Heikin Ashi典型价格
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    # 返回Heikin Ashi典型价格
    return Series(index=bars.index, data=res)

# 定义一个类，继承自IStrategy类
class ClucHAnix_5m(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """
    
    # 定义超参数
    buy_params = {
        "bbdelta_close": 0.01889,
        "bbdelta_tail": 0.72235,
        "close_bblower": 0.0127,
        "closedelta_close": 0.00916,
        "rocr_1h": 0.79492,
    }

    # Sell hyperspace params:
    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.99,
        "pPF_1": 0.02,
        "pPF_2": 0.05,
        "pSL_1": 0.02,
        "pSL_2": 0.04,

        # sell signal params
        'sell_fisher': 0.39075,
        'sell_bbmiddle_close': 0.99754
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.99  # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # buy params
    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize=True)
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize=True)
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize=True)
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize=True)
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize=True)

    # sell params
    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize=True)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize=True)

    # hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def informative_pairs(self):
        # 获取当前白名单中的交易对
        pairs = self.dp.current_whitelist()
        # 生成1小时时间框架的交易对列表
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    # come from BB_RPB_TSL
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        填充技术指标数据
        :param dataframe: 包含市场数据的DataFrame
        :param metadata: 包含交易对元数据的字典
        :return: 包含计算指标的DataFrame
        """
        # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'buy_tag'] = 'ClucHA'

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(self.rocr_1h.value)
            ) &
            ((
                     (dataframe['lower'].shift().gt(0)) &
                     (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                     (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                     (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                     (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                     (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
             ) |
             (
                     (dataframe['ha_close'] < dataframe['ema_slow']) &
                     (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
             )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['fisher'] > self.sell_fisher.value) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0),
            'sell'
        ] = 1

        return dataframe


class Cluc5mDCA(ClucHAnix_5m):
    position_adjustment_enable = True

    max_rebuy_orders = 1
    max_rebuy_multiplier = 2

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:

        if (self.config['position_adjustment_enable'] is True) and (self.config['stake_amount'] == 'unlimited'):
            return proposed_stake / self.max_rebuy_multiplier
        else:
            return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if (self.config['position_adjustment_enable'] is False) or (current_profit > -0.08):
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        # Maximum 2 rebuys, equal stake as the original
        if 0 < count_of_buys <= self.max_rebuy_orders:
            try:
                # This returns first order stake size
                stake_amount = filled_buys[0].cost
                # This then calculates current safety order size
                stake_amount = stake_amount
                return stake_amount
            except Exception as exception:
                return None

        return None
