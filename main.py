import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import baostock as bs
from matplotlib.ticker import MultipleLocator

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler


def find_stock(num):
    # 获取文件的信息
    path = '股票代码及对应公司名字.csv'
    stock_source = open(path)
    stock = pd.read_csv(stock_source)
    # pd.set_option('display.max_rows', None)  #显示所有的行
    mystock = ''
    myindex = -1
    for i in stock.code:
        matchObj = re.match(rf'.*{num}',i)
        myindex = myindex + 1
        if matchObj:
            mystock = matchObj.group()

            break

    #输出
    print("找到的股票的详细的信息")
    print(stock.iloc[myindex])
    # 后续再写用代码来找的
    # 自动找也没有办法找到剩下同类型的三支股票。那就算了

    # 自己找的
    # sh.600320   振华重工
    # sh.600031   三一重工
    # sz.300185   通裕重工
    # sz.300569   天能重工
    list = [mystock,'sh.600031', 'sz.300185', 'sz.300569']
    E_name = ['zhenghua', 'sanyi', 'tongyu', 'tianneng']
    C_name = ['振华重工', '三一重工', '通裕重工', '天能重工']

    data = [[list], [E_name], [C_name]]
    return data

'''
获取股票的数据
'''
def get_data(code):
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day).strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    # 登陆系统
    lg = bs.login()
    # 获取沪深A股历史K线数据
    rs_result = bs.query_history_k_data_plus(
            code,
            fields="date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3")
    df_result = rs_result.get_data()
    # 退出系统
    bs.logout()
    df_result['date'] = df_result['date'].map(
                        lambda x: datetime.strptime(x,'%Y-%m-%d'))
    _res = df_result.set_index('date')
    res = _res.applymap(lambda x: float(x))
    return res

'''
数据的处理
'''
def data_chuli(data):
    #获取数据,这几句运行一次就可以，数据就保存到本地的文件了
    # for name, code in zip(data[1][0], data[0][0]):
    #     exec(f"{name}=get_data(code)")
    #     exec(f"get_data(code).to_csv('{name}.csv')")

    for name,i in zip(data[1][0], range(4)):
        exec(f"data[1][0][i]= pd.read_csv('{name}.csv')")


    return data[1][0]

'''
获取月份刻度
'''
def get_mouth(new_data):
    # 自动获取月刻度
    for company in new_data:
        company.date = company.date.str.slice(0, 7)  # 切割月份
        data_need = company.date.drop_duplicates()  # 去重
        data_key = data_need.keys()    #获取索引
        # 获取所需月份的索引
        data_key1 = []
        if (data_key[0]<10):
            for i in range(int(len(data_key)/2)):
                data_key1.append(data_key[2*i+1])
        else:
            for i in range(int(len(data_key)/2)):
                data_key1.append(data_key[2*i])
        # 获取月份名称
        date_mouth = []
        for i in data_key1:
            date_mouth.append(data_need[i])

        return date_mouth

'''

'''
def lsspj(data, new_data, date_mouth):

    plt.figure(figsize=(15, 6))
    plt.suptitle('历史收盘价图', fontsize=30, color='blue')
    plt.subplots_adjust(top=1.25, bottom=1.2)
    for i, company in enumerate(new_data, 0):
        plt.subplot(2, 2, i+1)
        plt.ylabel('close')
        plt.xlabel(None)
        # 修改下面一句代码，结合前面的提示，将图里的英文标题改成中文标题
        plt.title(f"{data[2][0][i]}收盘价")
        # x轴刻度的
        company['close'].plot()
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15, 242, step=42), date_mouth, rotation=45)

    plt.tight_layout()
    plt.show()

'''

'''
def mrjyl(data, new_data, date_mouth):

    plt.figure(figsize=(15, 6))
    plt.suptitle('每日交易量图', fontsize=30, color='blue')
    plt.subplots_adjust(top=1.25, bottom=1.2)
    for i, company in enumerate(new_data, 0):
        plt.subplot(2, 2, i+1)
        plt.ylabel('volume')
        plt.xlabel(None)
        # 修改下面一句代码，结合前面的提示，将图里的英文标题改成中文标题
        plt.title(f"{data[2][0][i]}成交量")
        # x轴刻度的
        company['volume'].plot()
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15,242,step=42),date_mouth,rotation=45)
    plt.tight_layout()
    plt.show()

'''
'''
def MigAeage(data, new_data, date_mouth):
    ma_day = [10, 20, 50]
    for ma in ma_day:
        for company in new_data:
            column_name = f"MA for {ma} days"
            company[column_name] = company['close'].rolling(ma).mean()
    # 现在继续绘制所有额外的移动平均线。
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('平均移动线图', fontsize=30, color='blue')
    fig.set_figheight(12)
    fig.set_figwidth(18)
    # 修改斜体字部分，改用循环实现
    for i, company in enumerate(new_data, 0):
        if(i<2):
            company[['close',
                     'MA for 10 days',
                     'MA for 20 days',
                     'MA for 50 days']].plot(ax=axes[0, i])
            axes[0, i].set_title(f'{data[2][0][i-1]}移动平均线')
        else:
            company[['close',
                     'MA for 10 days',
                     'MA for 20 days',
                     'MA for 50 days']].plot(ax=axes[1, i-2])
            axes[1, i-2].set_title(f'{data[2][0][i - 1]}移动平均线')
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15, 242, step=42), date_mouth, rotation=45)

    fig.tight_layout()
    plt.show()

def pjrhbl(data, new_data, date_mouth):
    for company in new_data:
        company['Daily Return'] = company['close'].pct_change()
    # 画出日收益率
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.suptitle('平均日回报率图', fontsize=30, color='blue')
    for i, company in enumerate(new_data, 0):
        if (i<2):
            company['Daily Return'].plot(ax=axes[0, i],
                                         legend=True,
                                        linestyle='--',
                                         marker='o')
            axes[0, i].set_title(f'{data[2][0][i - 1]}平均日回报率')
        else:
            company['Daily Return'].plot(ax=axes[1, i-2],
                                         legend=True,
                                         linestyle='--',
                                         marker='o')
            axes[1, i - 2].set_title(f'{data[2][0][i - 1]}平均日回报率')
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15, 242, step=42), date_mouth, rotation=45)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(12, 7))
    plt.suptitle('平均日回报率直方图', fontsize=30, color='blue')
    for i, company in enumerate(new_data, 0):
        plt.subplot(2, 2, i+1)
        sns.histplot(company['Daily Return'].dropna(), bins=100, color='purple' ,kde=True)

        plt.ylabel('Daily Return')
        plt.title(f'{data[2][0][i]} 日回报率')
        # 也可以这样绘制
        # company['Daily Return'].dropna().hist(bins=100, color='purple')
        # sns.kdeplot(company['Daily Return'].dropna(), color='purple')

    plt.tight_layout()
    plt.show()


'''
回报率相关的图片
'''
def hbl(data, new_data):
    closing_df = pd.DataFrame()
    for company, company_n in zip(new_data, data[2][0]):
        temp_df = pd.DataFrame(index=company.index,
                               data = company['close'].values,
                               columns = [company_n])
        closing_df = pd.concat([closing_df,temp_df],axis = 1)
    print('所有的收盘价')
    print(closing_df.head())

    liquor_rets = closing_df.pct_change()
    print('股票的日回报')
    print(liquor_rets.head())

    # 单支日收益相关图
    hbl_dzrsyxg(liquor_rets)
    # 四支日收益相关图
    hbl_szrsyxg(liquor_rets, closing_df)
    # 相关图
    hbl_xgt(liquor_rets, closing_df)

def hbl_dzrsyxg(liquor_rets):
# # 1
    p1 = sns.jointplot(liquor_rets,
                  x = f'{data[2][0][0]}',
                  y = f'{data[2][0][0]}',
                  kind = 'scatter',
                  color ='seagreen',
                  )
    p1.fig.set_size_inches(10, 10)
    p1.fig.suptitle(f"{data[2][0][0]}自身的日收益相关性")
    plt.show()
    plt.close()

# 2
    p2 = sns.jointplot(liquor_rets,
                      x=f"{data[2][0][0]}",
                      y=f"{data[2][0][1]}",
                      kind='scatter',
                      color='seagreen',
                  )
    p2.fig.set_size_inches(10, 10)
    p2.fig.suptitle(f"{data[2][0][0]}与{data[2][0][1]}的日收益相关性")
    plt.show()
    plt.close()


# ++++++++++++++3+++++++++++++++++++
def hbl_szrsyxg(liquor_rets,closing_df):
    # 分析四支股票日收益相关性（pairplot+采用默认图形属性）
    p3 = sns.pairplot(liquor_rets, kind='reg')
    p3.fig.set_size_inches(20, 20)
    p3.fig.suptitle("四支股票的日收益相关性分析")
    plt.show()
    plt.close()

    # 通过命名为returns_fig来设置我们的图形，
    # 在DataFrame上调用PairPLot
    # 下面是分析四支股票日收益相关性(kde+散点图+直方图)
    return_fig = sns.PairGrid(liquor_rets.dropna())
    # 使用map_upper，我们可以指定上面的三角形是什么样的。
    # 可以对return_fig调用fig.suptitle()函数设置标题。
    return_fig.map_upper(plt.scatter, color='purple')
    # 我们还可以定义图中较低的三角形，
    # 包括绘图类型(kde)或颜色映射(blueppurple)
    return_fig.map_lower(sns.kdeplot, cmap='cool_d')
    # 最后，我们将把对角线定义为每日收益的一系列直方图
    return_fig.map_diag(plt.hist, bins=30)
    return_fig.fig.set_size_inches(14, 14)
    return_fig.fig.suptitle("四支股票日收益相关性(kde+散点图+直方图)")
    plt.show()
    plt.close()


    # 下面是分析四支股票收盘价相关性(kde+散点图+直方图)
    returns_fig = sns.PairGrid(closing_df)
    # 可以对return_fig调用fig.suptitle()函数设置标题。
    returns_fig.map_upper(plt.scatter, color='purple')
    returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
    returns_fig.map_diag(plt.hist, bins=30)
    returns_fig.fig.set_size_inches(15, 15)
    returns_fig.fig.suptitle("四支股票收盘价相关性(kde+散点图+直方图)")
    plt.show()
    plt.close()


def hbl_xgt(liquor_rets, closing_df):
    fig, ax = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(15)
    plt.suptitle('平均日回报率图', fontsize=30, color='blue')
    # 日回报的快速相关图
    sns.heatmap(liquor_rets.corr(),
                ax=ax[0],
                annot=True,
                cmap='summer')
    ax[0].set_title('日回报的快速相关图')
    # 每日收盘价的快速相关图
    sns.heatmap(closing_df.corr(),
                ax=ax[1],
                annot=True, cmap='summer')
    ax[1].set_title('每日收盘价的快速相关图')
    fig.tight_layout()
    plt.show()

    rets = liquor_rets.dropna()
    area = np.pi * 20
    plt.figure(figsize=(15, 10))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('预期回报', fontsize=18)
    plt.ylabel('风险', fontsize=18)
    plt.title('预期回报与风险相关图')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        if label == f'{data[2][0][3]}':  # 最后一只股票
            xytext = (-50, -50)
        else:
            xytext = (50, 50)
        plt.annotate(label, xy=(x, y), xytext=xytext,
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontsize=15,
                     arrowprops=dict(arrowstyle='->',
                                     color='gray',
                                     connectionstyle='arc3,rad=-0.3'))
    plt.show()
    plt.close()


def spjyc(mydata, new_data, date_mouth):
    for times, company in enumerate(new_data, 0):
        df = company.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        df.head()

        plt.figure(figsize=(13, 8))
        plt.title(f'{mydata[2][0][times]}历史收盘价', fontsize=20)
        plt.plot(df['close'])
        plt.xlabel('日期', fontsize=18)
        plt.ylabel('收盘价 RMB ('+ r"$\yen$"+')', fontsize=18)
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15, 242, step=42), date_mouth, rotation=45)
        plt.show()
        plt.close()

        # 创建一个只有收盘价的新数据帧
        data = df.filter(['close'])
        # 将数据帧转换为numpy数组
        dataset = data.values
        # 获取要对模型进行训练的行数
        training_data_len = int(np.ceil(len(dataset) * .95))
        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        # 创建训练集，训练标准化训练集
        train_data = scaled_data[0:int(training_data_len), :]
        # 将数据拆分为x_train和y_train数据集
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
            if i <= 61:
                print(x_train)
                print(y_train)
                # 将x_train和y_train转换为numpy数组
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape数据
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # 使用LSTM模型预测股价
        # pip install keras
        # 建立LSTM模型
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')
        # 训练模型
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # 创建测试数据集
        # 创建一个新的数组，包含从索引的缩放值
        test_data = scaled_data[training_data_len - 60:, :]
        # 创建数据集x_test和y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
            # 将数据转换为numpy数组
        x_test = np.array(x_test)
        # 重塑的数据
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # 得到模型的预测值
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        # 得到均方根误差(RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        # 将训练数据、实际数据集预测数据可视化。
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        print(f"使用LSTM模型预测{mydata[2][0][times]}股票的股价是")
        plt.figure(figsize=(13, 8))
        plt.title(f'{mydata[2][0][times]}模型')
        plt.xlabel('日期', fontsize=18)
        plt.ylabel('收盘价 RMB ('+r'$\yen$'+')', fontsize=18)
        plt.plot(train['close'])
        plt.plot(valid[['close', 'Predictions']])
        plt.legend(['训练价格', '实际价格', '预测价格'], loc='upper right')
        plt.gca().xaxis.set_major_locator(MultipleLocator(42))
        plt.xticks(np.arange(15, 242, step=42), date_mouth, rotation=45)
        plt.show()




if __name__ == "__main__":
    sns.set_style('whitegrid')
    plt.style.use("fivethirtyeight")
    # 修改此处，让其图中能正常显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False



    # 数据获取
    num = 320

    data = find_stock(num)
    # 处理数据，这里面有一段注释，是第一次运行必须要用的，
    # 详情见具体的函数
    new_data = data_chuli(data)


    # 画图需要的刻度值
    # 手动的
    # date_mouth = ['2021-12', '2022-02', '2022-04', '2022-06', '2022-08', '2022-10', ]
    # 自动的
    date_mouth =get_mouth(new_data)

    # 画出收盘价格
    # lsspj(data, new_data, date_mouth)
    # 画出每日交易量
    # mrjyl(data, new_data, date_mouth)
    # 各股票移动平均线
    # MigAeage(data, new_data, date_mouth)
    # 平均日回报率
    # pjrhbl(data, new_data, date_mouth)
    #股票的回报率
    hbl(data, new_data)

    # 收盘价预测
    # spjyc(data, new_data, date_mouth)


    # 将四支股票数据进行纵向合并
    # df = pd.concat(new_data,axis=0)





