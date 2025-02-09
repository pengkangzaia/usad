import numpy as np

from utils.eval_methods import *
from data.SWaT.swat_processor import *
from data.ServerMachineDataset.smd_processor import *
# from model.statefulsf_mogai import *
# from model.ae import *
from model.lstmvae import *

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


def smd_cal_all():
    # entity_ids = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8",
    #               "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
    #               "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    entity_ids = ["1-1", "1-2"]
    best_f1s = []
    for entity_id in entity_ids:
        smd_data = SMD(entity_id=entity_id, batch_size=BATCH_SIZE, window_size=window_size)
        train_loader, val_loader, test_loader = smd_data.get_dataloader()
        labels = smd_data.attack_labels

        # model = StatefulSfLinear(BATCH_SIZE, window_size, smd_data.input_feature_dim, hidden_size, latent_size,
        #            ensemble_size=4)
        model = LSTMVAE(BATCH_SIZE, window_size, smd_data.input_feature_dim, hidden_size, latent_size)
        # model = AE(window_size * smd_data.input_feature_dim, window_size * latent_size)
        input_shape = smd_data.input_feature_dim
        # model = BaggingAE(input_dim=input_shape, n_estimators=7, max_features=5, encoding_depth=2, decoding_depth=3, latent_dim=4)
        # model = UsadModel(window_size * smd_data.input_feature_dim, window_size * latent_size)
        # model = SF(BATCH_SIZE, window_size,smd_data.input_feature_dim, hidden_size, latent_size, num_layers=2, ensemble_size=4)
        model = to_device(model, device)

        val_loss = training(N_EPOCHS, model, train_loader, val_loader)
        # plot_simple_history(val_loss)
        # plot_train_loss(train_loss)
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
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        t, th = bf_search(y_pred, y_test, start=y_pred.min(), end=y_pred.max(), step_num=1000, display_freq=50)
        best_f1s.append(t[0])
    return best_f1s


best_f1s = smd_cal_all()
print(best_f1s)
print(np.mean(np.array(best_f1s)))

# t, th = bf_search(y_pred, y_test, start=0, end=1, step_num=1000, display_freq=50)
