import torch
from dataset.LstmAttention.make_data import class_split_point, cls2vec



def val_test_new(model, test_loader, enc_scale_list, enc_scale_value, dec_scale_list, dec_scale_value):
    global LOOKFORWARD
    global AIRPORT
    global device
    curr_cls_num = class_split_point[AIRPORT]
    model.eval()
    res = [[] for _ in range(LOOKFORWARD)]
    for enc_in, dec_in, target in test_loader:
            enc_in = feature_scale(enc_in, enc_scale_list, enc_scale_value)
            dec_in = feature_scale(dec_in, dec_scale_list, dec_scale_value)
            out = regressOut(enc_in, dec_in)
            for idx in range(LOOKFORWARD):
                out_argmax = torch.argmax(out[:, idx, :], dim = -1).to(device)
                target_argmax = torch.argmax(target[:, idx, :], dim = -1).to(device)
                out_argmax[out_argmax >= curr_cls_num] = curr_cls_num - 1
                target_argmax[target_argmax >= curr_cls_num] = curr_cls_num - 1
                mask = (out_argmax == target_argmax).to(torch.int32)
                correct_num = torch.sum(mask).item()
                res[idx].append(correct_num / len(mask))
    res = torch.tensor(res)
    res = torch.mean(res, dim=-1)
    for curr_res in res:
        print("{:.3f} ".format(curr_res), end="")
    print()
    
def val_test(model, test_loader, enc_scale_list, enc_scale_value, dec_scale_list, dec_scale_value):
    global AIRPORT
    global device
    curr_cls_num = class_split_point[AIRPORT]
    model.eval()
    res = 0
    for enc_in, dec_in, target in test_loader:
            enc_in = feature_scale(enc_in, enc_scale_list, enc_scale_value).to(device)
            dec_in = feature_scale(dec_in, dec_scale_list, dec_scale_value).to(device)
            out = regressOut(enc_in, dec_in)
            out = out.view(-1, out.shape[-1])
            target = target.view(-1, target.shape[-1])
            out_argmax = torch.argmax(out, dim = -1).to(device)
            target_argmax = torch.argmax(target, dim = -1).to(device)
            out_argmax[out_argmax >= curr_cls_num] = curr_cls_num - 1
            target_argmax[target_argmax >= curr_cls_num] = curr_cls_num - 1
            mask = (out_argmax == target_argmax).to(torch.int32)
            correct_num = torch.sum(mask).item()
            res += (correct_num / len(mask))

    return res / len(test_loader)    
    
    
def regressOut(model, enc_in, dec_in):
    global BATCH_SIZE
    global OUT_DIM
    global device
    res = torch.full((BATCH_SIZE, LOOKFORWARD, OUT_DIM), 0).float().to(device)
    dec_in[:, 1:, -20:] = 0
    for i in range(LOOKFORWARD):
        curr_res = model(enc_in, dec_in)
        batch_cls = torch.argmax(curr_res[:, i, :], dim = -1)
        res[list(range(BATCH_SIZE)), i, batch_cls] = 1.0
        if i < LOOKFORWARD - 1:
            dec_in[:, i+1, -20:] = torch.tensor([cls2vec[AIRPORT][idx.item()] for idx in batch_cls]).float().to(device)
    return res
        
    
    
def feature_scale(feature_batch, idx_mask, scale_values):
    _, _, fea_dim = feature_batch.shape
    scale_list = [1 for _ in range(fea_dim)]
    for i, value in enumerate(idx_mask):
        scale_list[value] = scale_values[i]
    mask = torch.tensor(scale_list, dtype=torch.float32).reshape(1, 1, -1)
    feature_batch = feature_batch * mask
    return feature_batch

