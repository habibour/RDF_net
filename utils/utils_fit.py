import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, gen, Epoch,
                  cuda, fp16, scaler, save_period, save_dir, local_rank=0,
                  method_name="baseline_pixel", alpha_feat=0.5, lambda_pixel=0.1, feat_warmup_epochs=10):
    """
    Train one epoch with dual training methods support.
    
    Args:
        method_name: "baseline_pixel" or "ours_feature"
        alpha_feat: Weight for feature loss
        lambda_pixel: Weight for pixel loss
        feat_warmup_epochs: Warmup epochs for feature loss
    """
    loss_total_sum = 0
    loss_det_sum = 0
    loss_pixel_sum = 0
    loss_feat_sum = 0
    criterion = nn.MSELoss()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    model_train.train()
    
    # Print method info at start of epoch 0
    if epoch == 0 and local_rank == 0:
        print(f"\n🎯 Training Method: {method_name}")
        print(f"   Lambda pixel: {lambda_pixel}")
        print(f"   Alpha feat: {alpha_feat}")
        print(f"   Feat warmup epochs: {feat_warmup_epochs}")

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        images, targets, clean = batch[0], batch[1], batch[2]
        
        # Assertions for fail-fast debugging
        assert images.shape == clean.shape, f"Images shape mismatch: {images.shape} vs {clean.shape}"
        
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                clean = clean.cuda(local_rank)
                hazy_and_clear = torch.cat([images, clean], dim=0)
            else:
                hazy_and_clear = torch.cat([images, clean], dim=0)
        
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
                
                # Extract features from restored image (dehazing output)
                _, restored_feats = model_train(outputs[3], return_feats=True, det_only=True)
                
                # Extract features from clean image (with no_grad for efficiency)
                with torch.no_grad():
                    _, clean_feats = model_train(clean, return_feats=True, det_only=True)
                
                # Assertions for feature matching
                assert len(restored_feats) == len(clean_feats), f"Feature count mismatch: {len(restored_feats)} vs {len(clean_feats)}"
                assert len(restored_feats) == 3, f"Expected 3 detector features, got {len(restored_feats)}"
                
                for i in range(len(restored_feats)):
                    assert restored_feats[i].shape == clean_feats[i].shape, f"Feature {i} shape mismatch: {restored_feats[i].shape} vs {clean_feats[i].shape}"
                
                # Feature-level L1 loss across all feature levels
                loss_feat = sum(torch.nn.functional.l1_loss(restored_feats[i], clean_feats[i].detach()) 
                               for i in range(len(restored_feats)))
                
                # Warmup schedule for feature loss
                warmup = 1.0
                if feat_warmup_epochs and feat_warmup_epochs > 0:
                    warmup = min(1.0, float(epoch + 1) / float(feat_warmup_epochs))
                
                loss_total = loss_det + (alpha_feat * warmup) * loss_feat
                
            else:
                loss_total = loss_det

            # Print detailed loss info for first iteration of first epoch
            if epoch == 0 and iteration == 0 and local_rank == 0:
                print(f"\n📊 Loss breakdown (epoch 0, iter 0):")
                print(f"   Detection loss: {loss_det.item():.6f}")
                print(f"   Pixel loss: {loss_pixel.item():.6f}")
                print(f"   Feature loss: {loss_feat.item():.6f}")
                if method_name == "ours_feature":
                    print(f"   Warmup factor: {warmup:.3f}")
                print(f"   Total loss: {loss_total.item():.6f}")

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
                    
                    # Extract features from restored image (dehazing output)
                    _, restored_feats = model_train(outputs[3], return_feats=True, det_only=True)
                    
                    # Extract features from clean image (with no_grad for efficiency)
                    with torch.no_grad():
                        _, clean_feats = model_train(clean, return_feats=True, det_only=True)
                    
                    # Assertions for feature matching
                    assert len(restored_feats) == len(clean_feats), f"Feature count mismatch: {len(restored_feats)} vs {len(clean_feats)}"
                    assert len(restored_feats) == 3, f"Expected 3 detector features, got {len(restored_feats)}"
                    
                    for i in range(len(restored_feats)):
                        assert restored_feats[i].shape == clean_feats[i].shape, f"Feature {i} shape mismatch: {restored_feats[i].shape} vs {clean_feats[i].shape}"
                    
                    # Feature-level L1 loss across all feature levels
                    loss_feat = sum(torch.nn.functional.l1_loss(restored_feats[i], clean_feats[i].detach()) 
                                   for i in range(len(restored_feats)))
                    
                    # Warmup schedule for feature loss
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
            pbar.set_postfix(**{
                'loss': loss_total_sum / (iteration + 1),
                'loss_detection': loss_det_sum / (iteration + 1),
                'pixel_loss': loss_pixel_sum / (iteration + 1),
                'feat_loss': loss_feat_sum / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    if local_rank == 0:
        pbar.close()
        
        # Log losses
        loss_history.append_loss(epoch + 1, loss_total_sum / epoch_step)
        loss_history.writer.add_scalar('loss_detection', loss_det_sum / epoch_step, epoch + 1)
        loss_history.writer.add_scalar('pixel_loss', loss_pixel_sum / epoch_step, epoch + 1)
        loss_history.writer.add_scalar('feat_loss', loss_feat_sum / epoch_step, epoch + 1)
        
        # Evaluation callback
        if eval_callback:
            eval_callback.on_epoch_end(epoch + 1, model_train_eval)
            
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss_total_sum / epoch_step:.6f}')
        print(f'Detection Loss: {loss_det_sum / epoch_step:.6f}')
        print(f'Pixel Loss: {loss_pixel_sum / epoch_step:.6f}')
        print(f'Feature Loss: {loss_feat_sum / epoch_step:.6f}')
        
        # Save model
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
            
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_path = os.path.join(save_dir, f"ep{epoch + 1:03d}-loss{loss_total_sum / epoch_step:.3f}.pth")
            torch.save(save_state_dict, save_path)
            
        if loss_total_sum / epoch_step <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        # End of training epoch
