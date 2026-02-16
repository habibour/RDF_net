import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, gen, Epoch,
                  cuda, fp16, scaler, save_period, save_dir, local_rank=0,
                  method_name="baseline_pixel", alpha_feat=0.5, lambda_pixel=0.1, feat_warmup_epochs=10):
    loss_total_sum = 0
    loss_det_sum = 0
    loss_pixel_sum = 0
    loss_feat_sum = 0
    criterion = nn.MSELoss()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, clean = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                clean = clean.cuda(local_rank)
                hazy_and_clear = torch.cat([images, clean], dim = 0).cuda()
            else:
                hazy_and_clear = torch.cat([images, clean], dim = 0)
        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(hazy_and_clear)
            detect_outputs = [outputs[0], outputs[1], outputs[2]]
            loss_det = yolo_loss(detect_outputs, targets, images)
            loss_pixel = torch.zeros_like(loss_det)
            loss_feat = torch.zeros_like(loss_det)

            if method_name == "baseline_pixel":
                if len(outputs) < 4:
                    raise RuntimeError("Dehazing output missing for baseline_pixel loss.")
                loss_pixel = criterion(outputs[3], clean)
                loss_total = loss_det + lambda_pixel * loss_pixel
            elif method_name == "ours_feature":
                if len(outputs) < 4:
                    raise RuntimeError("Dehazing output missing for ours_feature loss.")
                _, restored_feats = model_train(outputs[3], return_feats=True, det_only=True)
                with torch.no_grad():
                    _, clean_feats = model_train(clean, return_feats=True, det_only=True)
                loss_feat = sum(F.l1_loss(restored_feats[i], clean_feats[i]) for i in range(len(restored_feats)))
                warmup = 1.0
                if feat_warmup_epochs and feat_warmup_epochs > 0:
                    warmup = min(1.0, float(epoch + 1) / float(feat_warmup_epochs))
                loss_total = loss_det + (alpha_feat * warmup) * loss_feat
            else:
                loss_total = loss_det

            loss_total.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(hazy_and_clear)
                detect_outputs = [outputs[0], outputs[1], outputs[2]]
                loss_det = yolo_loss(detect_outputs, targets, images)
                loss_pixel = torch.zeros_like(loss_det)
                loss_feat = torch.zeros_like(loss_det)

                if method_name == "baseline_pixel":
                    if len(outputs) < 4:
                        raise RuntimeError("Dehazing output missing for baseline_pixel loss.")
                    loss_pixel = criterion(outputs[3], clean)
                    loss_total = loss_det + lambda_pixel * loss_pixel
                elif method_name == "ours_feature":
                    if len(outputs) < 4:
                        raise RuntimeError("Dehazing output missing for ours_feature loss.")
                    _, restored_feats = model_train(outputs[3], return_feats=True, det_only=True)
                    with torch.no_grad():
                        _, clean_feats = model_train(clean, return_feats=True, det_only=True)
                    loss_feat = sum(F.l1_loss(restored_feats[i], clean_feats[i]) for i in range(len(restored_feats)))
                    warmup = 1.0
                    if feat_warmup_epochs and feat_warmup_epochs > 0:
                        warmup = min(1.0, float(epoch + 1) / float(feat_warmup_epochs))
                    loss_total = loss_det + (alpha_feat * warmup) * loss_feat
                else:
                    loss_total = loss_det

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        loss_total_sum += loss_total.item()
        loss_det_sum += loss_det.item()
        loss_pixel_sum += loss_pixel.item()
        loss_feat_sum += loss_feat.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss_total_sum / (iteration + 1),
                                'loss_detection'  : loss_det_sum / (iteration + 1),
                                'pixel_loss': loss_pixel_sum / (iteration + 1),
                                'feat_loss': loss_feat_sum / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss_total_sum / epoch_step)
        loss_history.writer.add_scalar('loss_detection', loss_det_sum / epoch_step, epoch + 1)
        loss_history.writer.add_scalar('pixel_loss', loss_pixel_sum / epoch_step, epoch + 1)
        loss_history.writer.add_scalar('feat_loss', loss_feat_sum / epoch_step, epoch + 1)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (loss_total_sum / epoch_step))
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss_total_sum / epoch_step)))
        if loss_total_sum / epoch_step <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
