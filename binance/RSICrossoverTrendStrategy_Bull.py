import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy

class RSICrossoverTrendStrategy_Bull(IStrategy):
    """
    这是一个基于RSI指标的趋势跟踪策略，适用于币安交易所。
    该策略在RSI低于30时买入，在RSI高于70时卖出。
    """
    # 策略参数
    minimal_roi = {
        "0": 0.1  # 无止损，盈利10%时卖出
    }

    stoploss = -0.1  # 止损点设置为-10%

    timeframe = '1h'  # 使用1小时K线数据

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算策略所需的指标。
        :param dataframe: 包含历史价格数据的DataFrame
        :param metadata: 策略元数据
        :return: 包含计算指标的DataFrame
        """
        # 计算RSI指标，周期为14
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入信号。
        :param dataframe: 包含历史价格数据和计算指标的DataFrame
        :param metadata: 策略元数据
        :return: 包含买入信号的DataFrame
        """
        # 当RSI低于30时生成买入信号
        dataframe.loc[
            (
                (dataframe['rsi'] < 30)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成卖出信号。
        :param dataframe: 包含历史价格数据和计算指标的DataFrame
        :param metadata: 策略元数据
        :return: 包含卖出信号的DataFrame
        """
        # 当RSI高于70时生成卖出信号
        dataframe.loc[
            (
                (dataframe['rsi'] > 70)
            ),
            'sell'] = 1
        return dataframe