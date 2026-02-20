import os
import torch
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, 
                  method_name="baseline_pixel", alpha_feat=0.5, lambda_pixel=0.1, feat_warmup_epochs=10):
    """
    Training function for one epoch with dual loss methods
    
    Args:
        method_name: "baseline_pixel" or "ours_feature"
        alpha_feat: Feature loss weight (for ours_feature method)
        lambda_pixel: Pixel loss weight (for baseline_pixel method) 
        feat_warmup_epochs: Warmup epochs for feature loss
    """
    
    loss_total_sum     = 0
    loss_det_sum       = 0
    loss_pixel_sum     = 0
    loss_feat_sum      = 0
    
    val_loss = 0
    
    if local_rank == 0:
        print(f"Training with method: {method_name}")
        print(f"   Alpha feat: {alpha_feat}")
        print(f"   Lambda pixel: {lambda_pixel}")
        print(f"   Feat warmup epochs: {feat_warmup_epochs}")
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets, clean_images = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images      = images.cuda(local_rank)
                targets     = targets.cuda(local_rank)
                clean_images = clean_images.cuda(local_rank)
        
        optimizer.zero_grad()
        if not fp16:
            # Forward pass
            outputs = model_train(images)
            
            # Extract predictions from dict if needed
            if isinstance(outputs, dict):
                predictions = outputs.get('predictions', outputs.get('pred', outputs))
            else:
                predictions = outputs
            
            # Detection loss
            loss_det = yolo_loss(predictions, targets, images)
            
            # Method-specific loss computation
            if method_name == "baseline_pixel":
                # Pixel-level MSE loss between restored and clean images
                restored = model_train(images, return_feats=False)  # Get restored image
                if isinstance(restored, tuple):
                    restored = restored[0]  # Take first element if tuple
                loss_pixel = torch.nn.functional.mse_loss(restored, clean_images)
                loss_feat = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                
                loss_total = loss_det + lambda_pixel * loss_pixel
                
            elif method_name == "ours_feature":
                # Get features for loss computation
                with torch.no_grad():
                    clean_feats = model(clean_images, return_feats=True)
                    if not isinstance(clean_feats, (list, tuple)):
                        clean_feats = [clean_feats]
                
                # Get restored features
                restored_feats = model_train(images, return_feats=True)
                if not isinstance(restored_feats, (list, tuple)):
                    restored_feats = [restored_feats]
                
                # Feature-level L1 loss across all feature levels
                loss_feat = sum(torch.nn.functional.l1_loss(restored_feats[i], clean_feats[i].detach()) 
                               for i in range(len(restored_feats)))
                
                # Warmup schedule for feature loss
                warmup = 1.0
                if feat_warmup_epochs and feat_warmup_epochs > 0:
                    warmup = min(1.0, float(epoch + 1) / float(feat_warmup_epochs))
                
                loss_total = loss_det + (alpha_feat * warmup) * loss_feat
                loss_pixel = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                
            else:
                loss_total = loss_det
                loss_pixel = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                loss_feat = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)

            # Print detailed loss info for first iteration of first epoch
            if epoch == 0 and iteration == 0 and local_rank == 0:
                print(f"Method: {method_name}")
                print(f"Detection Loss: {loss_det.item():.6f}")
                print(f"Pixel Loss: {loss_pixel.item():.6f}")
                print(f"Feature Loss: {loss_feat.item():.6f}")
                print(f"Total Loss: {loss_total.item():.6f}")
                if method_name == "ours_feature":
                    print(f"Warmup factor: {warmup:.4f}")

            loss_total.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model_train(images)
                
                # Extract predictions from dict if needed
                if isinstance(outputs, dict):
                    predictions = outputs.get('predictions', outputs.get('pred', outputs))
                else:
                    predictions = outputs
                
                # Detection loss
                loss_det = yolo_loss(predictions, targets, images)
                
                # Method-specific loss computation
                if method_name == "baseline_pixel":
                    # Pixel-level MSE loss between restored and clean images
                    restored = model_train(images, return_feats=False)  # Get restored image
                    if isinstance(restored, tuple):
                        restored = restored[0]  # Take first element if tuple
                    loss_pixel = torch.nn.functional.mse_loss(restored, clean_images)
                    loss_feat = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                    
                    loss_total = loss_det + lambda_pixel * loss_pixel
                    
                elif method_name == "ours_feature":
                    # Get features for loss computation
                    with torch.no_grad():
                        clean_feats = model(clean_images, return_feats=True)
                        if not isinstance(clean_feats, (list, tuple)):
                            clean_feats = [clean_feats]
                    
                    # Get restored features
                    restored_feats = model_train(images, return_feats=True)
                    if not isinstance(restored_feats, (list, tuple)):
                        restored_feats = [restored_feats]
                    
                    # Feature-level L1 loss across all feature levels
                    loss_feat = sum(torch.nn.functional.l1_loss(restored_feats[i], clean_feats[i].detach()) 
                                   for i in range(len(restored_feats)))
                    
                    # Warmup schedule for feature loss
                    warmup = 1.0
                    if feat_warmup_epochs and feat_warmup_epochs > 0:
                        warmup = min(1.0, float(epoch + 1) / float(feat_warmup_epochs))
                    
                    loss_total = loss_det + (alpha_feat * warmup) * loss_feat
                    loss_pixel = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                    
                else:
                    loss_total = loss_det
                    loss_pixel = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)
                    loss_feat = torch.tensor(0.0).cuda() if cuda else torch.tensor(0.0)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_total_sum += loss_total.item()
        loss_det_sum += loss_det.item()
        loss_pixel_sum += loss_pixel.item()
        loss_feat_sum += loss_feat.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss_det' : loss_det_sum / (iteration + 1),
                                'loss_pixel': loss_pixel_sum / (iteration + 1),
                                'loss_feat': loss_feat_sum / (iteration + 1),
                                'loss_total': loss_total_sum / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train(images)
            
            # Extract predictions from dict if needed
            if isinstance(outputs, dict):
                predictions = outputs.get('predictions', outputs.get('pred', outputs))
            else:
                predictions = outputs
            
            loss_value = yolo_loss(predictions, targets, images)

            val_loss += loss_value.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        loss_history.append_loss(epoch + 1, loss_total_sum / epoch_step, val_loss / epoch_step_val)
        
        # Update EMA
        if ema:
            ema.update(model_train)
            # Evaluate EMA model
            eval_inputs = {
                "model": ema.ema,
                "loss_history": loss_history,
                "epoch": epoch + 1,
            }
            # Store for potential reuse or callbacks
            model_train_eval = eval_inputs
        else:
            model_train_eval = {"model": model_train, "loss_history": loss_history, "epoch": epoch + 1}
        
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
