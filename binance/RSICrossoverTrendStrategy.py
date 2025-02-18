# 导入必要的库
from datetime import datetime, timedelta  # 用于处理日期和时间
import talib.abstract as ta              # 技术分析库，用于计算各种技术指标
import pandas_ta as pta                  # pandas技术分析扩展库
from freqtrade.persistence import Trade  # freqtrade交易对象
from freqtrade.strategy.interface import IStrategy  # freqtrade策略接口
from pandas import DataFrame             # 数据处理框架
from freqtrade.strategy import DecimalParameter, IntParameter  # 参数优化类
from functools import reduce            # 用于合并多个条件
import warnings                         # 警告处理

# 忽略运行时警告
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# 全局临时持仓列表
TMP_HOLD = []   # 用于跟踪MA120和MA240之上的交易
TMP_HOLD1 = []  # 用于跟踪开仓价格距离MA120较远的交易

class RSICrossoverTrendStrategy(IStrategy):
    # 最小ROI配置，这里设置为1表示100%的利润才会触发ROI卖出
    minimal_roi = {
        "0": 1
    }
    
    timeframe = '5m'                     # 使用5分钟K线
    process_only_new_candles = True      # 只在新K线形成时处理
    startup_candle_count = 240           # 启动需要的K线数量
    
    # 订单类型配置
    order_types = {
        'entry': 'market',               # 入场使用市价单
        'exit': 'market',                # 出场使用市价单
        'emergency_exit': 'market',       # 紧急出场使用市价单
        'force_entry': 'market',         # 强制入场使用市价单
        'force_exit': "market",          # 强制出场使用市价单
        'stoploss': 'market',            # 止损使用市价单
        'stoploss_on_exchange': False,    # 不在交易所设置止损
        'stoploss_on_exchange_interval': 60,  # 交易所止损检查间隔
        'stoploss_on_exchange_market_ratio': 0.99  # 交易所止损市场比率
    }

    # 止损和追踪止损配置
    stoploss = -0.25                     # 25%的固定止损
    trailing_stop = False                # 关闭追踪止损
    trailing_stop_positive = 0.002       # 正向追踪止损
    trailing_stop_positive_offset = 0.05  # 追踪止损偏移
    trailing_only_offset_is_reached = True  # 仅在达到偏移时启用追踪

    use_custom_stoploss = True           # 启用自定义止损

    # 买入参数优化配置
    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)

    # 卖出参数优化配置
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # CCI指标相关参数
    cci_opt = False
    sell_loss_cci = IntParameter(low=0, high=600, default=120, space='sell', optimize=cci_opt)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.05, decimals=2, space='sell', optimize=cci_opt)

    @property
    def protections(self):
        # 冷却期保护，防止频繁交易
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 18
            }
        ]
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # 自定义动态止损逻辑
        if current_profit >= 0.05:        # 当利润达到5%
            return -0.002                 # 设置0.2%的止损
            
        if str(trade.enter_tag) == "buy_new" and current_profit >= 0.03:  # 对于buy_new信号且利润达到3%
            return -0.003                 # 设置0.3%的止损

        return None                       # 其他情况使用默认止损

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算买入信号相关指标
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)  # 15周期简单移动平均
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)  # CTI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # 14周期RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)  # 4周期快速RSI
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)  # 20周期慢速RSI

        # 计算获利卖出相关指标
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)  # 随机快速指标
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)  # CCI指标

        # 计算移动平均线
        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)  # 120周期MA
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)  # 240周期MA

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''  # 初始化入场标签

        # 买入条件1
        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &  # RSI下降趋势
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &      # 快速RSI低于阈值
                (dataframe['rsi'] > self.buy_rsi_32.value) &                # RSI高于阈值
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &  # 价格低于SMA15
                (dataframe['cti'] < self.buy_cti_32.value)                  # CTI低于阈值
        )

        # 新买入条件
        buy_new = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &  # RSI下降趋势
                (dataframe['rsi_fast'] < 34) &                              # 快速RSI低于34
                (dataframe['rsi'] > 28) &                                   # RSI高于28
                (dataframe['close'] < dataframe['sma_15'] * 0.96) &         # 价格低于SMA15的96%
                (dataframe['cti'] < self.buy_cti_32.value)                  # CTI低于阈值
        )

        # 添加买入条件到条件列表
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_new)
        dataframe.loc[buy_new, 'enter_tag'] += 'buy_new'

        # 合并所有买入条件
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # 获取当前K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # 计算最大亏损
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        
        # 价格高于MA120和MA240时的处理
        if current_candle['close'] > current_candle["ma120"] and current_candle['close'] > current_candle["ma240"]:
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)
        
        # 开仓价格远高于MA120时的处理
        if (trade.open_rate - current_candle["ma120"]) / trade.open_rate >= 0.1:
            if trade.id not in TMP_HOLD1:
                TMP_HOLD1.append(trade.id)
        
        # 获利卖出条件
        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"
        
        # 止损卖出条件（基于CCI）
        if min_profit <= -0.1:
            if current_profit > self.sell_loss_cci_profit.value:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell"

        # MA120快速卖出条件
        if trade.id in TMP_HOLD1 and current_candle["close"] < current_candle["ma120"]:
            TMP_HOLD1.remove(trade.id)
            return "ma120_sell_fast"

        # MA120和MA240卖出条件
        if trade.id in TMP_HOLD and current_candle["close"] < current_candle["ma120"] and current_candle["close"] < \
                current_candle["ma240"]:
            if min_profit <= -0.1:
                TMP_HOLD.remove(trade.id)
                return "ma120_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 初始化卖出信号
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
