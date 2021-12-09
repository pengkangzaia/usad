from statefulsf import *
import torch.utils.data as data_utils
from sklearn import preprocessing
from eval_methods import *

device = get_default_device()

# Read data
# normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")  # , nrows=1000)
normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv", nrows=1000)  # , nrows=1000)
normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
# normal.shape

# Transform all columns into float64
for i in list(normal):
    normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
normal = normal.astype(float)

# 数据预处理
min_max_scaler = preprocessing.MinMaxScaler()
x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

# Read data
# attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv", sep=";")  # , nrows=1000)
attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv", sep=";", nrows=1000)  # , nrows=1000)
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

############## training ###################
# BATCH_SIZE = 7919
BATCH_SIZE = 200
N_EPOCHS = 5
hidden_size = 100
latent_size = 40

# w_size = windows_normal.shape[1] * windows_normal.shape[2]  # window_size * feature_size
# z_size = windows_normal.shape[1] * hidden_size  # window_size * hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0], windows_attack.shape[1], windows_attack.shape[2]]))
), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = StatefulSf(BATCH_SIZE, window_size, windows_normal.shape[2], hidden_size, latent_size, ensemble_size=window_size)
# model = to_device(model, device)
model.to_local_device()

val_loss, train_loss = training(N_EPOCHS, model, train_loader, val_loader)
plot_simple_history(val_loss)
plot_train_loss(train_loss)
torch.save({'ae': model.state_dict()}, "model.pth")

############ testing #################


checkpoint = torch.load("model.pth")

model.load_state_dict(checkpoint['ae'])

# 每一个batch都有一个result。组成result集合
results = testing(model, test_loader)
windows_labels = []
for i in range(len(labels) - window_size):
    windows_labels.append(list(np.int_(labels[i:i + window_size])))
# 窗口中有误差，则为异常，表示为1
y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
# 样本太少的话，误差会很大
y_pred = np.concatenate(
    [torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
     results[-1].flatten().detach().cpu().numpy()])
y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
threshold = ROC(y_test, y_pred)

t, th = bf_search(y_pred, y_test, start=0, end=1, step_num=1000, display_freq=50)

