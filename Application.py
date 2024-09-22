import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QVBoxLayout, QMainWindow, QWidget, QSizePolicy
from PyQt5 import uic, QtGui
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class SummaryDialog(QDialog):
    def __init__(self, summary_text, df, parent=None):
        super(SummaryDialog, self).__init__(parent)
        uic.loadUi("Results.ui", self)
        self.SummaryResult.setText(summary_text)
        self.setupMatplotlibCanvas()
        self.visualize_signals(df)
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.setWindowTitle("Backtesting Trading Strategy")

    def setupMatplotlibCanvas(self):
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        if hasattr(self, 'graphLayout'):
            while self.graphLayout.count():
                child = self.graphLayout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.graphLayout = QVBoxLayout(self.graphWidget)
        
        self.graphLayout.addWidget(self.canvas)

    def visualize_signals(self, df):
        self.canvas.axes.clear()
        self.canvas.axes.set_title('Close Price History with Buy & Sell Signals', fontsize=18)
        self.canvas.axes.plot(df['Close'], alpha=0.5, label='Close Price')

        marker_dict = {'Buy': '^', 'Short': 'v', 'Sell': 'o', 'Cover': '*'}
        for category in ['Buy', 'Short', 'Sell', 'Cover']:
            signal_data = df[df['Action'] == category]
            if not signal_data.empty:
                self.canvas.axes.scatter(signal_data.index, signal_data['Close'], marker=marker_dict[category], label=category)
        self.canvas.axes.set_xlabel('Date', fontsize=15)
        self.canvas.axes.set_ylabel('Close Price', fontsize=15)
        self.canvas.axes.legend()
        self.canvas.draw()  

class MyWindow(QWidget):
    def __ini__(self):
        super().__init_()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowIcon(QtGui.QIcon('icon.png'))

class Main(QMainWindow):          
    def __init__(self):
        super().__init__()
        self.init_ui("Ticker Chooser.ui")
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        
    def init_ui(self, ui_file):
        uic.loadUi(ui_file, self)
        self.Submit.clicked.connect(self.load_second_ui)
        self.show()

        
    def load_second_ui(self):
        self.stock_ticker = self.lineEdit_ticker.text()
        self.start_date = self.lineEdit_startdate.text()
        self.end_date = self.lineEdit_enddate.text()
        self.init_ui("Strategy Chooser.ui")
        self.prices_df = yf.download(self.stock_ticker, period='max')
        self.Backtest_button.clicked.connect(self.execute_indicators)
        self.Goback.clicked.connect(self.go_back_to_ticker_chooser)



    def go_back_to_ticker_chooser(self):
        self.init_ui("Ticker Chooser.ui")
    def execute_indicators(self):
        if self.checkBox_MR.isChecked():
            self.MR_function(self.prices_df)
        if self.checkBox_RSI.isChecked():
            self.rsi_function(self.prices_df)
        if self.checkBox_MACD.isChecked():
            self.macd_function(self.prices_df)
        if self.checkBox_Stochastic.isChecked():
            self.stochastic_function(self.prices_df)
        self.close()


    def show_summary(self, summary_text, df):
        self.summary_dialog = SummaryDialog(summary_text, df, self)
        self.summary_dialog.show()


    def go_back_to_strategy_chooser(self):
        self.summary_dialog.close()
        self.init_ui("Strategy Chooser.ui")
    
    def MR_function(self, df, period=30, column="Adj Close", buy_threshold=0.95, sell_threshold=1.05):
        print(f"Executing Mean Reversion Backtesting on {self.stock_ticker}...")
        df['SMA'] = df[column].rolling(window=period).mean() 
        df['Simple_Returns'] = df[column].pct_change(1)
        df['MR_Log_Returns'] = np.log(1 + df['Simple_Returns'])
        df['Ratio'] = df[column] / df['SMA']
        df['Positions'] = np.where(df['Ratio'] > sell_threshold, -1, np.nan)
        df['Positions'] = np.where(df['Ratio'] < buy_threshold, 1, df['Positions'])
        df['Positions'].ffill(inplace=True)
        df['MR_Signal'] = np.where(df['Positions'] == 1, 'Buy', np.nan)
        df['MR_Signal'] = np.where(df['Positions'] == -1, 'Sell', df['MR_Signal'])
        df['Buy'] = np.where(df['Positions'] == 1, df[column], np.nan)
        df['Sell'] = np.where(df['Positions'] == -1, df[column], np.nan)
        df['MR_Strategy_Returns'] = df['Positions'].shift(1) * df['MR_Log_Returns']
        print(f"Finished Executing Mean Reversion Backtesting on {self.stock_ticker}...")
        return df
        
    
    def rsi_function(self, df, window=14, oversold_lvl = 30, overbought_lvl = 70):
        print(f"Executing RSI Backtesting on {self.stock_ticker}...")
        diff = df['Close'].diff(1)
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Signal'] = 0

        for i in range(len(df)):
            if df['RSI'].iloc[i] > oversold_lvl:
                df.loc[df.index[i], 'RSI_Signal'] = 'Buy'
            elif df['RSI'].iloc[i] < overbought_lvl:
                df.loc[df.index[i], 'RSI_Signal'] = 'Sell'
            else:
                df.loc[df.index[i], 'RSI_Signal'] = 'Hold'

        print(f"Finished Executing RSI Backtesting on {self.stock_ticker}...")
    
        return df
        

    def macd_function(self, df, short_window=12, long_window=26, signal_window=9):
        print(f"Executing MACD Backtesting on {self.stock_ticker}...")
        df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['MACD_Signal'] = 0


        for i in range(len(df)):
            if df['MACD'].iloc[i] > df['MACD_Signal_Line'].iloc[i]:
                df.loc[df.index[i], 'MACD_Signal'] = 'Buy'
            elif df['MACD'].iloc[i] < df['MACD_Signal_Line'].iloc[i]:
                df.loc[df.index[i], 'MACD_Signal'] = 'Sell'
            else:
                df.loc[df.index[i], 'MACD_Signal'] = 'Hold'

        print(f"Finished Executing MACD Backtesting on {self.stock_ticker}...")
        
        return df
        
    def stochastic_function(self, df, k_window=14, d_window=3):
        print(f"Executing Stochastic Backtesting on {self.stock_ticker}...")
        low_min = df['Low'].rolling(window=k_window).min()
        high_max = df['High'].rolling(window=k_window).max()

        df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['%D'] = df['%K'].rolling(window=d_window).mean()
        df['Stochastic_Signal'] = 0

        for i in range(len(df)):
            if df['%K'].iloc[i] < df['%D'].iloc[i]:
                df.loc[df.index[i], 'Stochastic_Signal'] = 'Buy'
            elif df['%K'].iloc[i] > df['%D'].iloc[i]:
                df.loc[df.index[i], 'Stochastic_Signal']  = 'Sell'
            else:
                df.loc[df.index[i], 'Stochastic_Signal']  = 'Hold'

        print(f"Finished Executing Stochastic Backtesting on {self.stock_ticker}...")
        
        return df


    def execute_indicators(self):
        signal_list = []
        
        if self.checkBox_MR.isChecked():
            self.MR_function(self.prices_df)
            signal_list.append('MR_Signal')
            
        if self.checkBox_RSI.isChecked():
            self.rsi_function(self.prices_df)
            signal_list.append('RSI_Signal')
            
        if self.checkBox_MACD.isChecked():
            self.macd_function(self.prices_df)
            signal_list.append('MACD_Signal')
            
        if self.checkBox_Stochastic.isChecked():
            self.stochastic_function(self.prices_df)
            signal_list.append('Stochastic_Signal')
            
        self.backtest_strategy(self.prices_df, signal_list)


    def nearest_date(self, dates, target_date):
        nearest = min(dates, key=lambda x: abs(x - target_date))
        return nearest


    def backtest_strategy(self, df, signal_list, initial_capital=10000, trade_size=1):
        print("Backtesting...")
        
        df.index = pd.to_datetime(df.index)

        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')        
        available_dates = df.index.tolist()
        
        start_date_nearest = self.nearest_date(available_dates, start_date)
        end_date_nearest = self.nearest_date(available_dates, end_date)

        df = df.loc[start_date_nearest:end_date_nearest]

        df['Action'] = 0


        capital = initial_capital
        position = 0
        buys = sells = shorts = covers = wins = total_trades = 0
        entry_price = 0
        profit = 0

        for i in range(len(df)):

            is_buy = (df.iloc[i][signal_list] == 'Buy').all()
            is_sell = (df.iloc[i][signal_list] == 'Sell').all()

            close = df.iloc[i]['Close']

            if is_buy == True:
                if position < 0:
                    profit = (entry_price - close) * abs(position)
                    capital += profit
                    covers += 1
                    position = 0
                    df.at[df.index[i], 'Action'] = 'Cover'

                    if profit > 0:
                        wins += 1

                elif position == 0:
                    position += trade_size
                    capital -= close * trade_size
                    buys += 1
                    entry_price = close
                    df.at[df.index[i], 'Action'] = 'Buy'

                else:
                    continue

            elif is_sell == True:
                if position > 0:
                    profit = (close - entry_price) * abs(position)
                    capital += profit
                    sells += 1
                    position = 0
                    df.at[df.index[i], 'Action'] = 'Sell'

                    if profit > 0:
                        wins += 1

                elif position == 0:
                    position -= trade_size
                    capital += close * trade_size
                    shorts += 1
                    entry_price = close
                    df.at[df.index[i], 'Action'] = 'Short'

            else:
                continue

        total_trades = buys + sells + shorts + covers
        win_rate = (wins / (total_trades // 2) * 100) if total_trades != 0 else 0
        profit = capital - initial_capital
        buy_hold = (df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']

        backtest_summary = {
            "Initial Capital": f"${initial_capital:.2f}",
            "End Capital": f"${capital:.2f}",
            "Total Trades": total_trades,
            "Buys": buys,
            "Sells": sells,
            "Shorts": shorts,
            "Covers": covers,
            "Wins": wins,
            "Win Rate": f"{win_rate:.2f}%",
            "Outstanding Position": position,
            "Outstanding Position Value": f"${(position * df.iloc[-1]['Close']):.2f}",
            "Overall Profit": f"${profit:.2f}",
            "Buy & Hold Returns": f"{buy_hold:.2f}%"}

        backtest_summary = backtest_summary if 'backtest_summary' in locals() else "No trades executed"

        if isinstance(backtest_summary, dict):
            summary_text = "\n".join([f"{k}: {v}" for k, v in backtest_summary.items()])
        else:
            summary_text = backtest_summary

        self.show_summary(summary_text, df)
        print("Showing Results...")

        return backtest_summary


    def show_summary(self, summary_text, df):
        self.summary_dialog = SummaryDialog(summary_text, df, self)
        self.summary_dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())