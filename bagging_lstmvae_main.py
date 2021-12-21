import numpy as np

from model.bagging_lstmvae_pro import *
import torch.utils.data as data_utils
from utils.eval_methods import *
from sklearn import preprocessing

device = get_default_device()

min_max_scaler = preprocessing.MinMaxScaler()
# Read data
normal = pd.read_csv("data/SWaT/SWaT_Dataset_Normal_v1.csv", nrows=1000)  # , nrows=1000)
normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
# Transform all columns into float64
for i in list(normal):
    normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
normal = normal.astype(float)
# 数据归一化
x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

# Read data
attack = pd.read_csv("data/SWaT/SWaT_Dataset_Attack_v0.csv", sep=";", nrows=1000)  # , nrows=1000)
labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
# Transform all columns into float64
for i in list(attack):
    attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
attack = attack.astype(float)
x = attack.values
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)

############## windows ###################
window_size = 12
# np.arange(window_size)[None, :] 1*12 (0,1,2,3,4,5,6,7,8,9,10,11)一行12列
# np.arange(normal.shape[0] - window_size)[:, None] (1000-12)*1 (0,1,2,3,4,5...) 988列，每列递增
# np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None] (1000-12)*12
windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]
windows_attack = attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))
y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
y_test = np.array(y_test)
############## training ###################
BATCH_SIZE = 500
N_EPOCHS = 3
N = 5 * round((normal.shape[1] / 3) / 5)  # 10 for both bootstrap sample size and number of estimators
decoder_layers = 2  # number of hidden layers for each decoder
z = int((N / 2) - 1)  # size of latent space

windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(
        ([windows_normal_train.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(
        ([windows_normal_val.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(
        ([windows_attack.shape[0], windows_attack.shape[1], windows_attack.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = BaggingLstmVAE(time_step=window_size,
                       input_dim=normal.shape[1],
                       hidden_size=N,
                       n_estimators=N,
                       max_features=N,
                       latent_dim=z,
                       decoding_depth=decoder_layers)
for i in range(model.n_estimators):
    model.LSTMVAEs[i] = to_device(model.LSTMVAEs[i], device)
    model.DivLstmVAEs[i] = to_device(model.DivLstmVAEs[i], device)

history = training(N_EPOCHS, model, train_loader)

lower, upper = testing(model, test_loader)
# 点调整法
windows_attack = windows_attack[:, -1, :]
attack_tiles = np.tile(windows_attack.reshape(windows_attack.shape[0], 1, windows_attack.shape[1]), (1, N, 1))
result = np.where((attack_tiles < lower.numpy()) | (attack_tiles > upper.numpy()), 1, 0)
inference = np.mean(np.mean(result, axis=1), axis=1)
print(inference[0:100])
t, th = bf_search(inference, y_test, start=0, end=1, step_num=1000, display_freq=50)

result = np.where((attack_tiles < upper.numpy()) | (attack_tiles > lower.numpy()), 1, 0)
inference = np.mean(np.mean(result, axis=1), axis=1)
print(inference[0:100])
t, th = bf_search(inference, y_test, start=0, end=1, step_num=1000, display_freq=50)

a = lower.detach().cpu()[:,-1,:].numpy()
b = upper.detach().cpu()[:,-1,:].numpy()
for i in range(a.shape[-1]):
  print("========计算第" + str(i) + "个特征========")
  aUb, aLb, eq = [], [], []
  for j in range(len(a[:,i])):
      if a[j][i] > b[j][i]:
          aUb.append(j)
      elif a[j][i] < b[j][i]:
          aLb.append(j)
      else:
          eq.append(j)
  print("lower bound大于upper bound的个数为：" + str(len(aUb)))
  print("lower bound小于upper bound的个数为：" + str(len(aLb)))
  print("lower bound等于upper bound的个数为：" + str(len(eq)))
