import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



ETF_list = [
#  ["SPY", "SPY"],
#  ["TLT", "TLT"],
#  ["ks11", "KS11"],
#  ["IAU", "IAU"],
#  ["SCHP", "SCHP"],
  ["btc", "BTC/KRW"],
#  ["QQQ", "QQQ"]
]


def get_df(asset_list):
	df_list = [fdr.DataReader(code, '2008-01-01', '2020-11-30')['Close'] for name, code in asset_list]
	df = pd.concat(df_list, axis=1)
	df.columns = [name for name, code in asset_list]
	print(df)
	return df


BTC_df = get_df(ETF_list)

BTC_df = BTC_df.to_numpy()

train_data = BTC_df[:1000]
test_data = BTC_df[1000:]
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = train_data_normalized.cuda()


train_window = 10
def create_inout_sequences(input_data, tw):
	inout_seq = []
	L = len(input_data)
	for i in range(L-tw):
		train_seq = input_data[i:i+tw]
		train_label = input_data[i+tw:i+tw+1]
		inout_seq.append((train_seq ,train_label))
	return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)






class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size

		self.lstm = nn.LSTM(input_size, hidden_layer_size)

		self.linear = nn.Linear(hidden_layer_size, output_size)

		self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(),
							torch.zeros(1,1,self.hidden_layer_size).cuda())

	def forward(self, input_seq):
		lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
		predictions = self.linear(lstm_out.view(len(input_seq), -1))
		return predictions[-1]


model = LSTM()
model = model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)


epochs = 50

for i in range(epochs):
	for seq, labels in train_inout_seq:
		optimizer.zero_grad()
		model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
						torch.zeros(1, 1, model.hidden_layer_size).cuda())

		y_pred = model(seq)

		single_loss = loss_function(y_pred, labels)
		single_loss.backward()
		optimizer.step()

	print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 2
test_inputs = train_data_normalized[-train_window:].tolist()


model.eval()

for i in range(fut_pred):
	seq = torch.FloatTensor(test_inputs[-train_window:]).cuda()
	with torch.no_grad():
		model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
						torch.zeros(1, 1, model.hidden_layer_size).cuda())
		test_inputs.append(model(seq).item())




test_inputs[fut_pred:]
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))


print("Pred #############")
print(actual_predictions)
print("Last #############")
print(BTC_df[-3:])

x=np.array(['2020-12-19', '2020-12-20'], dtype='datetime64[D]')

plt.ylabel('BTC')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#plt.plot(BTC_df,label='DATa')
plt.plot(x,actual_predictions,'.',label='Prediction')
plt.legend()
plt.savefig("aa.png")







