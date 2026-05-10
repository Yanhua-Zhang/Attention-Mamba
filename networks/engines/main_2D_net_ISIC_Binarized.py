import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

# from torch.utils.data import DataLoader
from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.util import save_transformed_img_RGB
from data_loaders.dataset_loader_ISIC2018_2D import Dataset_ISIC2018_2D
from data_augmentations.augmentations_ISIC import get_augmentations




# -----------------------------------------------------------------------------
# for Training, Val, Test
def main_ISIC_2D(args, model, logger, writer, optimizer, lr_scheduler_train, lr_scheduler_warmup):
    
    # -------------------
    # training

    if args.if_training:

        logger.info("-----------------------------------------------------------------------------")

        # -------------------
        # load train set
        trainset_Trans = get_augmentations(args, logger, mode='train')

        logger.info("Start loading Train set: ")
        trainset = Dataset_ISIC2018_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='train', transform=trainset_Trans)
        logger.info("The length of train set is: {}".format(len(trainset)))
        logger.info("Finish loading Train set.")

        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

        max_iterations = args.max_epochs * len(trainloader)  
        logger.info("Iterations per epoch: {}. Total iterations: {}".format(len(trainloader), max_iterations))

        # for the val during training
        logger.info("Start loading Val sets: ")

        # valset_Trans = get_augmentations(args, logger, mode='val') # leads to Bug
        valset = Dataset_ISIC2018_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='test_ISIC2018', transform=None)
        logger.info("The length of Val set is: {}".format(len(valset)))
        logger.info("Finish loading Val sets.")

        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

        # -------------------
        iter_num = 0

        best_val_DSC = 0.0  # to find the best trained model
        best_val_DSC_epoch = 0.0  

        best_val_loss = 10000 # to find the best trained model
        best_val_loss_epoch = 0

        best_train_loss = 10000 # to find the best trained model
        best_train_loss_epoch = 0

        best_train_main_loss = 10000 # to find the best trained model
        best_train_main_loss_epoch = 0

        for epoch_num in range(args.max_epochs):

            model.train()

            logger.info('Starting epoch {}/{}'.format(epoch_num + 1, args.max_epochs))
            logger.info("Start Training: ")

            # -------------------
            # update scheduler
            if args.if_use_warmup:

                if epoch_num == 0:
                    lr_scheduler = lr_scheduler_warmup  
                    logger.info('Use warm up scheduler.')                  
                elif epoch_num == (args.warmup_epochs-1):
                    lr_scheduler = lr_scheduler_train
                    logger.info('Switch to training scheduler.')  

            elif epoch_num == 0:
                lr_scheduler = lr_scheduler_train
                logger.info('Use training scheduler.') 

            # -------------------
            # record lr   
            for param_group in optimizer.param_groups:
                epoch_lr = param_group['lr']

            writer.add_scalar('Train/epoch_lr', epoch_lr,  epoch_num + 1)
            logger.info('epoch %d: lr: %f' % (epoch_num + 1, epoch_lr))
            
            # -------------------
            epoch_loss = 0

            epoch_main_loss = 0  

            for i_batch, (dict_data) in enumerate(trainloader):
                
                img = dict_data['Img'].as_tensor().to(torch.float32).cuda()          # b, 3, y, x  
                label_GT = dict_data['GT'].as_tensor().cuda()      # b, 1, y, x

                # -------------------
                # save transformed imgs for checking
                if args.if_save_Transformed_img and ((i_batch % 200) == 0):
                    save_transformed_img_RGB(args, logger, i_batch, dict_data)

                # -------------------
                outputs = model(img) # must be a list
                
                # -------------------
                # calculate loss
                label_list = [label_GT]
                
                loss = args.calc_loss_fun(outputs, label_list)
                
                # -------------------
                optimizer.zero_grad()  # zero out existing gradients
                loss.backward()
                optimizer.step()  # update gradients

                # -------------------
                # record loss of each iter
                iter_num = iter_num + 1

                # exclude the aux losses
                main_loss = args.calc_loss_fun([outputs[0]], [label_GT])   # this is used for finding the best-trained model
                
                writer.add_scalar('Train/loss_iter', loss, iter_num)
                writer.add_scalar('Train/loss_main_iter', main_loss, iter_num)
                # logger.info('iteration %d: loss: %f' % (iter_num, loss.item()))
                
                epoch_loss += loss.item() # calculate sum loss of the iterations of this epoch
                epoch_main_loss += main_loss.item()  # this is used for finding the best-trained model

            # -------------------
            # renew 
            lr_scheduler.step()

            # -------------------
            # record epoch loss

            epoch_loss = epoch_loss / (i_batch + 1)

            writer.add_scalar('Train/loss_epoch', epoch_loss, epoch_num + 1)
            logger.info('epoch %d: loss: %f' % (epoch_num + 1, epoch_loss))

            epoch_main_loss = epoch_main_loss / (i_batch + 1)

            writer.add_scalar('Train/loss_main_epoch', epoch_main_loss, epoch_num + 1)
            logger.info('epoch %d: loss_main: %f' % (epoch_num + 1, epoch_main_loss))

            # -------------------
            # save epoch model + Delete the last model

            save_model_path = os.path.join(args.Model_path, 'epoch_' + str(epoch_num + 1) + '.pth')
            torch.save(model.state_dict(), save_model_path)
            logger.info("save current model to {}".format(save_model_path))

            last_save_model_path = os.path.join(args.Model_path, 'epoch_' + str(epoch_num) + '.pth')

            if os.path.exists(last_save_model_path):
                os.remove(last_save_model_path)
                logger.info("Deleted last model: {}".format(last_save_model_path))

            # -------------------
            # val 

            if (epoch_num % args.val_interval) == 0:

                logger.info("Start Validation: ")

                # resize img for fast inference
                val_loss, val_DSC = val_2D_Resize(args, logger, model, valloader)

                logger.info('epoch %d: Val loss: %f' % (epoch_num + 1, val_loss))
                logger.info('epoch %d: Val DSC: %f' % (epoch_num + 1, val_DSC))

                writer.add_scalar('Val/loss_epoch', val_loss, epoch_num + 1)
                writer.add_scalar('Val/DSC_epoch', val_DSC, epoch_num + 1)

                logger.info("Finished Training & Validation.")

            else:
                logger.info("Skip Validation...")

            # -------------------
            # save best val model

            if (epoch_num % args.val_interval) == 0:
                # according to val acc
                if val_DSC > best_val_DSC:
                    best_val_DSC = val_DSC
                    best_val_DSC_epoch = epoch_num
                    save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_val_acc.pth')
                    torch.save(model.state_dict(), save_model_best_path)
                    logger.info("According to val acc, find best trained model at epoch: {}, save best model to: {}".format(epoch_num + 1, save_model_best_path))

                # according to val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_epoch = epoch_num
                    save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_val_loss.pth')
                    torch.save(model.state_dict(), save_model_best_path)
                    logger.info("According to val loss, find best trained model at epoch: {}, save best model to: {}".format(epoch_num + 1, save_model_best_path))

            # according to train loss
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                best_train_loss_epoch = epoch_num
                save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_train_loss.pth')
                torch.save(model.state_dict(), save_model_best_path)
                logger.info("According to train loss, find best trained model at epoch: {}, save best model to: {}".format(epoch_num + 1, save_model_best_path))

            # according to train main loss
            if epoch_main_loss < best_train_main_loss:
                best_train_main_loss = epoch_main_loss
                best_train_main_loss_epoch = epoch_num
                save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_train_main_loss.pth')
                torch.save(model.state_dict(), save_model_best_path)
                logger.info("According to train main loss, find best trained model at epoch: {}, save best model to: {}".format(epoch_num + 1, save_model_best_path))

        writer.close()

        logger.info("During the whole training, find the trained model with the best val acc at epoch: {}, the best val acc is: {}".format(best_val_DSC_epoch + 1, best_val_DSC))

        logger.info("During the whole training, find the trained model with the lowest val loss at epoch: {}, the lowest val loss is: {}".format(best_val_loss_epoch + 1, best_val_loss))

        logger.info("During the whole training, find the trained model with the lowest train loss at epoch: {}, the lowest train loss is: {}".format(best_train_loss_epoch + 1, best_train_loss))

        logger.info("During the whole training, find the trained model with the lowest train main loss at epoch: {}, the lowest train main loss is: {}".format(best_train_main_loss_epoch + 1, best_train_main_loss))

        logger.info("-----------------------------------------------------------------------------")


    # -------------------
    # Inference: Test set
    if args.if_inferece_testset:

        logger.info("-----------------------------------------------------------------------------")
        logger.info("Start inference on different Test set: ...")

        # -------------------
        # load best trained model

        if args.load_model_type == 'load_best_val_loss_model':
            save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_val_loss.pth')
            logger.info("load best val loss model for inference")

        elif args.load_model_type == 'load_best_val_acc_model':
            save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_val_acc.pth')
            logger.info("load best val acc model for inference")

        elif args.load_model_type == 'load_best_train_loss_model':
            save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_train_loss.pth')
            logger.info("load best train loss model for inference")

        elif args.load_model_type == 'load_best_train_main_loss_model':
            save_model_best_path = os.path.join(args.Model_best_path, 'best_trained_model_train_main_loss.pth')
            logger.info("load best train main loss model for inference")

        elif args.load_model_type == 'load_best_the_last_model':
            save_model_best_path = os.path.join(args.Model_path, 'epoch_' + str(args.max_epochs) + '.pth')
            logger.info("load the last model for inference")

        else:
            raise ValueError('This load type is not supported: ' + args.load_model_type)  

        if not os.path.exists(save_model_best_path):
            raise ValueError('best model file is lost: ' + str(save_model_best_path))
        else:
            logger.info("load best model for test set inference, from path: {} ".format(save_model_best_path))


        # model.load_state_dict(torch.load(save_model_best_path))

        checkpoint = torch.load(save_model_best_path)

        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        logger.info("Missing keys (Weights your model needs but aren't in file: {}".format(str(list(missing)[:5])))
        logger.info("Unexpected keys (Weights in file but not in your model): {}".format(str(list(unexpected)[:5])))

        # Create a new state dict excluding the thop/profiling keys
        clean_state_dict = {k: v for k, v in checkpoint.items() if "total_ops" not in k and "total_params" not in k}

        # remove 'total_ops' and 'total_params'
        for name, module in model.named_modules():
            if hasattr(module, 'total_ops'):
                del module.total_ops
            if hasattr(module, 'total_params'):
                del module.total_params

        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(clean_state_dict.keys())
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        logger.info("Missing keys (Weights your model needs but aren't in file: {}".format(str(list(missing)[:5])))
        logger.info("Unexpected keys (Weights in file but not in your model): {}".format(str(list(unexpected)[:5])))

        # Load the cleaned version
        model.load_state_dict(clean_state_dict, strict=True)


        # -------------------
        # inference on different testset

        save_path_tests = [args.Softmax_test_ISIC2018_path, args.Softmax_test_PH2_path, args.Softmax_test_all_path]

        index = 0

        for mode_test in ['test_ISIC2018', 'test_PH2', 'test_all']:

            # -------------------
            # load test set
            logger.info("Start loading Test set: {}".format(mode_test))
            testset = Dataset_ISIC2018_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode=mode_test, transform=None)
            logger.info("The length of Test set is: {}".format(len(testset)))
            logger.info("Finish loading Test set.")

            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

            # -------------------
            # inference

            inference_2D_Resize_MS_Flipping(args, model, logger, datasetloader = testloader, save_path = save_path_tests[index])
            logger.info("Inference type: {}".format('Resize img for fast inference'))

            # -------------------
            logger.info("Finished inference on the Test set: {}".format(mode_test))
            index += 1

        logger.info("Finished inference on the Test set.")


from tools_RGB.utils_metrics_Binarized import cal_Dice, cal_Dice_V2, cal_Dice_V3
# -----------------------------------------------------------------------------
# for validation

def val_2D_Resize(args, logger, model, valloader):

    model.eval()
    
    loss_sum = 0

    dice_sum = 0

    with torch.no_grad():

        for i_batch, (dict_data) in enumerate(valloader):
            
            img = dict_data['Img'].cuda()          # b, 3, y, x 
            label_GT = dict_data['GT'].cuda()      # b, 1, y, x 

            # this operation is necessary, as we use bilinear for GT resize
            label_GT_binarized = torch.where(label_GT >= 0.5, 1.0, 0.0)

            # -------------------
            x_size = img.size()       # b, 3, y, x

            # b, 3, y, x ---> b, 3, y', x'
            img_resized = F.interpolate(img, args.img_size, mode='bilinear', align_corners=False)

            logits = model(img_resized)

            # logger.info("logits:" + str(logits.size()))

            # b, 1, y', x' ---> b, 1, y, x
            logits_up = F.interpolate(logits, x_size[2:], mode='bilinear', align_corners=False)

            # logger.info("logits_up:" + str(logits_up.size()))

            # -------------------
            # calculate loss

            loss = args.calc_loss_fun([logits_up], [label_GT])

            loss_sum += loss.item()

            # -------------------
            # calculate Dice

            dice = cal_Dice_V2(logger = logger, pre_logits = logits_up, gt_tensor = label_GT_binarized)

            dice_sum += dice


    # -------------------
    # acc
    avg_dice = dice_sum / (i_batch + 1)

    # -------------------
    # loss
    avg_loss = loss_sum / (i_batch + 1)

    return avg_loss, avg_dice


# -----------------------------------------------------------------------------
# resize img for fast inference

def inference_2D_Resize_MS_Flipping(args, model, logger, datasetloader, save_path):

    model.eval()

    dice_sum = 0

    with torch.no_grad():

        for i_batch, (dict_data) in enumerate(datasetloader):
            
            img = dict_data['Img'].cuda()         # 1, 3, y, x 
            label_GT = dict_data['GT'].cuda()     # 1, 1, y, x 
            file_name = dict_data['file_name']

            # this operation is necessary, as we use bilinear for GT resize
            label_GT_binarized = torch.where(label_GT >= 0.5, 1.0, 0.0)

            # -------------------
            x_size = img.size()       # 1, 3, y, x

            # 1, 3, y, x ---> 1, 3, y', x'
            img_resized = F.interpolate(img, args.img_size, mode='bilinear', align_corners=False)

            # -------------------
            # zoom img for MS pred:

            ori_z, _, ori_y, ori_x = img_resized.size()  # 1, 3, y', x'

            final_sigmoid = torch.zeros([ori_z, args.num_classes, ori_y, ori_x]).cuda()   # 1, 1, y', x'

            if 1 not in args.inference_scales:
                raise ValueError('1 must be in inference_scales.')
            

            for scale in args.inference_scales:
                
                # -------------------
                # scale, if necessary
                if scale != 1:
                    scaled_input = F.interpolate(img_resized, scale_factor=(scale, scale), mode='area')  # 1, 3, y', x' ---> 1, 3, y'', x''

                elif scale == 1:
                    scaled_input = img_resized   # 1, 3, y'', x''

                # -------------------
                # flipping inference

                if args.if_inf_flip:

                    # -------------------
                    # slide window inf
                    logits_scaled = model(scaled_input)  # 1, 1, y'', x''
                    
                    pred_sigmoid = logits_scaled.sigmoid()  # 1, 1, y'', x''

                    # -------------------
                    # horizontally, flip

                    logits_scaled = model(torch.flip(scaled_input,  dims=[3]))  # 1, 1, y'', x''

                    pred_sigmoid += torch.flip(logits_scaled, dims=[3]).sigmoid() # horizontally, flip back: 1, 1, y'', x''

                    # -------------------
                    # vertically, flip

                    logits_scaled = model(torch.flip(scaled_input,  dims=[2]))  # 1, 1, y'', x''

                    pred_sigmoid += torch.flip(logits_scaled, dims=[2]).sigmoid()  # vertically, flip back: 1, 1, y'', x''

                    # -------------------
                    # vertically + horizontally

                    logits_scaled = model(torch.flip(scaled_input,  dims=[2, 3]))  # 1, 1, y'', x''

                    pred_sigmoid += torch.flip(logits_scaled, dims=[2, 3]).sigmoid()  # vertically, flip back: 1, 1, y'', x''

                    # -------------------
                    # do avg
                    pred_sigmoid = pred_sigmoid / 4  # 1, 1, y'', x''

                    logger.info("Do advanced flip for MS inference: horizontally, vertically, horizontally + vertically, nonlin ---> sum ---> avg")

                else:

                    logits_scaled = model(scaled_input)  # 1, 1, y'', x''

                    pred_sigmoid = logits_scaled.sigmoid()  # 1, 1, y'', x''

                    logger.info("Not do flip for MS inference")
                
                # -------------------
                # scale back, if necessary
                if scale != 1:
                    scaled_sigmoid = F.interpolate(pred_sigmoid, size=(ori_y, ori_x), mode='area')    # 1, 1, y', x'

                elif scale == 1:
                    scaled_sigmoid = pred_sigmoid      # 1, 1, y', x'

                assert final_sigmoid.size() == scaled_sigmoid.size(), "must keep the same shape, after MS inference"

                final_sigmoid += scaled_sigmoid   # 1, 1, y', x'

            # -------------------
            pred_softmax = final_sigmoid / len(args.inference_scales) # 1, 1, y', x'

            # 1, 1, y', x' ---> 1, 1, y, x
            pred_softmax_up = F.interpolate(pred_softmax, x_size[2:], mode='bilinear', align_corners=False)

            if i_batch % 30 == 0:
                # logger.info('softmax pred size: ' + str(pred_softmax_np.size))
                logger.info('pred_softmax_up shape: ' + str(pred_softmax_up.size()))
                logger.info('pred_softmax_up unique: ' + str(torch.unique(pred_softmax_up)))

            # -------------------
            # calculate Dice

            dice = cal_Dice_V3(logger = logger, pre_sigmoid = pred_softmax_up, gt_tensor = label_GT_binarized)

            dice_sum += dice

            # -------------------
            # save softmax predictions

            pred_softmax_np = pred_softmax_up.squeeze(0).cpu().numpy().astype(np.float32) # 1, 1, y, x ---> 1, y, x ---> cpu ---> np ---> float32

            if i_batch % 30 == 0:
                # logger.info('softmax pred size: ' + str(pred_softmax_np.size))
                logger.info('softmax pred shape: ' + str(pred_softmax_np.shape))
                logger.info('softmax pred unique: ' + str(np.unique(pred_softmax_np)))
                # logger.info('softmax pred ndim: ' + str(pred_softmax_np.ndim))
                # logger.info('softmax pred type: ' + str(type(pred_softmax_np)))
                # logger.info('softmax pred dtype: ' + str(pred_softmax_np.dtype))

            # npz_path = os.path.join(save_path, file_name[0] + '.npz')
            # np.savez(npz_path, softmax = pred_softmax_np)
            # logger.info("save softmax prediction to this path: {} ".format(npz_path))

            npy_path = os.path.join(save_path, file_name[0] + '.npy')
            np.save(npy_path, pred_softmax_np)
            logger.info("save softmax prediction to this path: {} ".format(npy_path))

        # -------------------
        # acc
        avg_dice = dice_sum / (i_batch + 1)

        logger.info('The avg dice of this test set is: ' + str(avg_dice))
            
