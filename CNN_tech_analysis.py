import os
import warnings
from datetime import datetime, date
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import chardet

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


class DataConstruct():
    def __init__(self, data_path):
        self.data_path = data_path
        self.stock_data = self.process_data()

    def concat_data(self):
        all_data = pd.DataFrame()
        csv_list = os.listdir(self.data_path)
        csv_list = [csv for csv in csv_list if csv.endswith('.csv')]
        sample_name = csv_list[0][:-5]
        #enc = detect_encoding(data_path+csv_list[0])
        for i in range(len(csv_list)):
            file_name = sample_name + str(i+1) + '.csv'
            df = pd.read_csv(self.data_path+file_name, encoding='gb2312') #enc.lower()
            all_data = pd.concat([all_data,df], ignore_index=True)
        #all_data.to_csv(sample_name + '.csv', encoding='latin1', index=False)
        return all_data

    def process_data(self):
        stock_data = self.concat_data()
        stock_data.drop(columns=stock_data.columns[-1], inplace=True)
        stock_data.rename(columns={col: col.split('_')[-1] for col in stock_data.columns}, inplace=True)
        stock_data['Stkcd'] = stock_data['Stkcd'].astype(str).str.zfill(6)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
        stock_data.set_index(['Stkcd', 'Date'], inplace=True)
        stock_data.dropna(how='all', inplace=True)
        #print(stock_data.index.levels[0].tolist())
        #print(stock_data.index.levels[0].tolist().index('000725'))
        for stock_code in stock_data.index.levels[0].tolist():
            try:
                stock_data.loc[stock_code].interpolate(method='time', inplace=True)
            except KeyError:
                continue
        return stock_data

    def get_tag_data(self, image_len, predict_len, height):
        image_data = []
        tag_data = []
        for stock_code in self.stock_data.index.levels[0].tolist():
            try:
                this_stock_data = self.stock_data.loc[stock_code]
            except KeyError:
                continue
            date_list = this_stock_data.index.tolist()
            this_stock_data['ClprNext'] = this_stock_data['Clpr'].shift(-predict_len)
            this_stock_data['IfRet'] = (this_stock_data['ClprNext'] > this_stock_data['Clpr']).astype(int)
            #tag_data += this_stock_data['IfRet'].tolist()
            for i in range(0, len(date_list)-predict_len, image_len):
                ohlc_data = this_stock_data.iloc[i:i+image_len, :4]
                ohlc_data = ohlc_data.values
                try:
                    image_size = (3 * image_len, height)
                    image_data.append(generate_ohlc_image(ohlc_data, image_size))
                    tag_data.append(this_stock_data['IfRet'][i])
                except ValueError:
                    print('股票: ' + stock_code + ' 日期: ' + datetime.strftime(date_list[i], '%Y-%m-%d'))
                    print(ohlc_data)
                    continue
        return image_data, tag_data


def generate_ohlc_image(ohlc_data, image_size):
    # 获取最高价、最低价、开盘价、收盘价数据
    high_prices = ohlc_data[:, 1]
    low_prices = ohlc_data[:, 2]
    open_prices = ohlc_data[:, 0]
    close_prices = ohlc_data[:, 3]
    # 计算价格范围
    price_range = np.max(high_prices) - np.min(low_prices)
    #image_size=(3*len(ohlc_data), 100) #int(price_range/0.01)
    # 创建图像数据数组
    image_data = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    # 映射OHLC数据到图像像素
    for i in range(len(ohlc_data)):
        high_pixel = int((high_prices[i] - np.min(low_prices)) / price_range * (image_size[1] - 1))
        low_pixel = int((low_prices[i] - np.min(low_prices)) / price_range * (image_size[1] - 1))
        open_pixel = int((open_prices[i] - np.min(low_prices)) / price_range * (image_size[1] - 1))
        close_pixel = int((close_prices[i] - np.min(low_prices)) / price_range * (image_size[1] - 1))
        # 在图像中绘制矩形
        image_data[low_pixel:high_pixel+1, 3*i+1] = 255  # 绘制最高价和最低价之间的线段
        image_data[open_pixel, 3*i] = 255
        image_data[close_pixel, 3*i+2] = 255
        #image_data[min(open_pixel, close_pixel):max(open_pixel, close_pixel) + 1, i] = 255  # 绘制开盘价和收盘价之间的线段
    # 创建PIL图像对象
    image_data = np.flip(image_data, axis=0)
    #print(image_data)
    image = Image.fromarray(image_data)
    return image

'''def get_ohlc(stock_data, stock_code, time_interval, start_time):
    start_time1 = datetime.strptime(start_time, "%Y-%m-%d")
    this_stock_data = stock_data.loc[stock_code]
    date_list = this_stock_data.index.tolist()
    try:
        ohlc_data = this_stock_data.iloc[date_list.index(start_time1):date_list.index(start_time1)+time_interval, 0:4]
        #print(ohlc_data)
        ohlc_data = ohlc_data.values
        image = generate_ohlc_image(ohlc_data)
        image.show()
        return image
    except ValueError:
        print("找不到对应时间的数据！")
        return 0'''


class CNNModel:
    def __init__(self, image_len, predict_len, constructed_data):
        self.model = models.Sequential()
        self.cnn_init()
        self.image_len = image_len
        self.predict_len = predict_len
        self.height = 100
        self.constructed_data = constructed_data
        #self.image_data, self.tag_data = constructed_data.get_tag_data(self.image_len, self.predict_len, self.height)

    def get_train_test(self, train_ratio=0.7):
        image_data, tag_data = self.constructed_data.get_tag_data(self.image_len, self.predict_len, self.height)
        X = tf.constant(image_data)
        y = tf.constant(tag_data)
        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_size = int(len(self.dataset) * train_ratio)
        dataset_shuffled = self.dataset.shuffle(buffer_size=len(self.dataset))
        #X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), train_size=train_size, test_size=1-train_size)
        #self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        #self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        train_dataset = dataset_shuffled.take(train_size).batch(batch_size=32).repeat()
        test_dataset = dataset_shuffled.skip(train_size).batch(batch_size=32).repeat()
        print("训练集大小:", train_size)
        print("测试集大小:", len(self.dataset)-train_size)
        return train_dataset, test_dataset

    def cnn_init(self):
        self.model.add(layers.Input(shape=(self.height, 3*self.image_len, 1)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

    def cnn_summary(self):
        self.model.summary()

    def cnn_train(self):
        train_dataset, test_dataset = self.get_train_test()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(train_dataset, epochs=10, validation_data=test_dataset)
        return history

    def cnn_evaluate(self):
        history = self.cnn_train()
        loss, accuracy = self.model.evaluate(history.validation_data)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)
        return loss, accuracy

stock_data = DataConstruct('data_resset_1990/')
#image = get_ohlc(stock_data, '000001', 20, '1994-01-05')
#print(np.array(image))
image_data, tag_data = stock_data.get_tag_data(60, 60)
print(str(len(image_data)) + ' ' + str(len(tag_data)))
n = np.random.randint(0, len(image_data))
image_data[n].show()
