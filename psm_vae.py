# 准备工作
from model.bagging_vae import *
import torch.utils.data as data_utils
from utils.eval_methods import *
from sklearn import preprocessing

x_train = pd.read_csv('data/PSM/train.csv', index_col=[0], nrows=10000)
x_train.fillna(0, inplace=True)  # imputes missing values
x_test = pd.read_csv('data/PSM/test.csv', index_col=[0])
y_test = pd.read_csv('data/PSM/test_label.csv', index_col=[0])
t_train = np.tile(x_train.index.values.reshape(-1,1), (1, x_train.shape[1]))
t_test = np.tile(x_test.index.values.reshape(-1,1), (1, x_train.shape[1]))
from sklearn.preprocessing import MinMaxScaler
xscaler = MinMaxScaler()
x_train_scaled = xscaler.fit_transform(x_train.values)
x_test_scaled = xscaler.transform(x_test.values)

N = 5 * round((x_train.shape[1] / 3) / 5)  # 10 for both bootstrap sample size and number of estimators
encoder_layers = 1  # number of hidden layers for each encoder
decoder_layers = 2  # number of hidden layers for each decoder
z = int((N / 2) - 1)  # size of latent space
activation = 'relu'
output_activation = 'sigmoid'
S = 5  # Number of frequency components to fit to input signals
delta = 0.05
freq_warmup = 5  # pre-training epochs
sin_warmup = 5  # synchronization pre-training
BATCH_SIZE = 180



train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(x_train_scaled).float()
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
#     torch.from_numpy(windows_normal_val).float()
# ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(x_test_scaled).float()
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = BaggingVAE(input_dim=x_train.shape[1], n_estimators=N, max_features=N, encoding_depth=encoder_layers,
                  decoding_depth=decoder_layers, latent_dim=z)


for i in range(model.n_estimators):
  model.VAEs[i] = to_device(model.VAEs[i], device)
  model.DivVAEs[i] = to_device(model.DivVAEs[i], device)

history = training(N, model, train_loader)
y_test = y_test['label'].values

# 测试
lower, upper = testing(model, test_loader)
synched = x_test.values
# 点调整法
attack_tiles = np.tile(synched.reshape(synched.shape[0], 1, synched.shape[1]), (1, N, 1))
result = np.where((attack_tiles < lower.cpu().numpy()) | (attack_tiles > upper.cpu().numpy()), 1, 0)
inference = np.mean(np.mean(result, axis=1), axis=1)

print(inference[0:100])

t, th = bf_search(inference, y_test, start=0.1, end=0.99, step_num=int((0.99-0.1)/0.01), display_freq=50)