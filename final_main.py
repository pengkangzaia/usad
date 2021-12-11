from model.statefulsf import *
from utils.eval_methods import *
from data.SWaT.swat_processor import *
from data.ServerMachineDataset.smd_processor import *

device = get_default_device()

############## training ###################
# BATCH_SIZE = 7919
BATCH_SIZE = 200
N_EPOCHS = 5
hidden_size = 60
latent_size = 30
window_size = 12

# w_size = windows_normal.shape[1] * windows_normal.shape[2]  # window_size * feature_size
# z_size = windows_normal.shape[1] * hidden_size  # window_size * hidden_size

# swat_data = SWaT(BATCH_SIZE, window_size=window_size, read_rows=1000)
# train_loader, val_loader, test_loader = swat_data.get_dataloader()
# labels = swat_data.attack_labels


smd_data = SMD(entity_id="1-1", batch_size=BATCH_SIZE, window_size=window_size)
train_loader, val_loader, test_loader= smd_data.get_dataloader()
labels = smd_data.attack_labels

model = StatefulSf(BATCH_SIZE, window_size, smd_data.input_feature_dim, hidden_size, latent_size, ensemble_size=window_size)
# model = to_device(model, device)
model.to_local_device()

val_loss, train_loss = training(N_EPOCHS, model, train_loader, val_loader)
plot_simple_history(val_loss)
plot_train_loss(train_loss)
torch.save({'ae': model.state_dict()}, "saved_model/model.pth")

############ testing #################


checkpoint = torch.load("saved_model/model.pth")

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
threshold = ROC(y_test, y_pred)

t, th = bf_search(y_pred, y_test, start=0, end=1, step_num=1000, display_freq=50)
