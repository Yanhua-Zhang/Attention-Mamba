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

from data_loaders.dataset_loader_ACDC_2D import Dataset_ACDC_2D
from data_augmentations.augmentations_ACDC import get_augmentations

# from torch.utils.data import DataLoader
from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.util import save_transformed_img_2D_One_modality, save_img_3D




# -----------------------------------------------------------------------------
# for Training, Val, Test
def main_ACDC_2D(args, model, logger, writer, optimizer, lr_scheduler_train, lr_scheduler_warmup):
    
    # -------------------
    # training

    if args.if_training:

        logger.info("-----------------------------------------------------------------------------")

        # -------------------
        # load train set
        trainset_Trans = get_augmentations(args, logger, mode='train')

        logger.info("Start loading Train set: ")
        trainset = Dataset_ACDC_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='Train', fold=args.fold_name, transform=trainset_Trans)
        logger.info("The length of train set is: {}".format(len(trainset)))
        logger.info("Finish loading Train set.")

        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

        max_iterations = args.max_epochs * len(trainloader)  
        logger.info("Iterations per epoch: {}. Total iterations: {}".format(len(trainloader), max_iterations))

        # for the val during training
        logger.info("Start loading Val sets: ")

        # valset_Trans = get_augmentations(args, logger, mode='val') # leads to Bug
        valset = Dataset_ACDC_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='Val', fold=args.fold_name, transform=None)
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
                
                img = dict_data['Img'].as_tensor().to(torch.float32).cuda()          # b, 1, y, x  
                label_GT = dict_data['GT'].as_tensor().long().cuda()      # b, 1, y, x

                # -------------------
                # save transformed imgs for checking
                if args.if_save_Transformed_img and ((i_batch % 200) == 0):
                    save_transformed_img_2D_One_modality(args, logger, i_batch, dict_data)

                # logger.info('img shape: ' + str(img.shape))
                # logger.info('label_GT shape: ' + str(label_GT.shape))

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

                # use sliding window with overlap=0.0, so that no need to do spatial resize
                val_loss, val_DSC = val_2D_Slide_Window(args, model, valloader)

                logger.info('epoch %d: Val loss: %f' % (epoch_num + 1, val_loss))
                logger.info('epoch %d: Val DSC: %f' % (epoch_num + 1, val_DSC))

                writer.add_scalar('Val/loss_epoch', val_loss, epoch_num + 1)
                writer.add_scalar('Val/DSC_epoch', val_DSC, epoch_num + 1)

                logger.info("Finished Training & Validation.")

            else:
                logger.info("Skip Validation: ")

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
    # Inference: Val set

    if args.if_inferece_valset:

        logger.info("-----------------------------------------------------------------------------")
        logger.info("Start inference on the Val set: ...")

        # -------------------
        # load val set
        logger.info("Start loading Val sets: ")
        valset = Dataset_ACDC_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='Val', fold=args.fold_name, transform=None)
        logger.info("The length of Val set is: {}".format(len(valset)))
        logger.info("Finish loading Val sets.")

        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

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
            logger.info("load best model for val set inference, from path: {}".format(save_model_best_path))

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
        # inference

        if args.inference_type == 'MS_inference':

            inference_2D_MS_Slide_Window(args, model, logger, datasetloader = valloader, save_path = args.Softmax_val_path)
            logger.info("Inference type: {}".format(args.inference_type))

        else:
            raise ValueError('This inference type is not supported.')

        logger.info("Finished inference on the Val set.")

    # -------------------
    # Inference: Test set
    if args.if_inferece_testset:

        logger.info("-----------------------------------------------------------------------------")
        logger.info("Start inference on the Test set: ...")

        # -------------------
        # load test set
        logger.info("Start loading Test set: ")
        testset = Dataset_ACDC_2D(logger, args, path_dataset=args.path_dataset, path_split=args.path_split, mode='Test', fold=args.fold_name, transform=None)
        logger.info("The length of Test set is: {}".format(len(testset)))
        logger.info("Finish loading Test set.")

        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

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
        # inference

        if args.inference_type == 'MS_inference':

            inference_2D_MS_Slide_Window(args, model, logger, datasetloader = testloader, save_path = args.Softmax_test_path)
            logger.info("Inference type: {}".format(args.inference_type))

        else:
            raise ValueError('This inference type is not supported.')

        logger.info("Finished inference on the Test set.")


# -----------------------------------------------------------------------------
# for validation

# use sliding window with overlap=0.0, so that no need to do spatial resize
def val_2D_Slide_Window(args, model, valloader):

    model.eval()
    
    loss_sum = 0

    # exclude the first category (channel index 0)
    # num_classes: always including the background
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=args.num_classes)  

    with torch.no_grad():

        for i_batch, (dict_data) in enumerate(valloader):
            
            img = dict_data['Img'].permute(1, 0, 2, 3).cuda()          # 1, z, y, x ---> z, 1, y, x
            label_GT = dict_data['GT'].squeeze(0).cuda()               # 1, z, y, x ---> z, y, x

            # use sliding window with overlap=0.0, so that no need to do spatial resize
            ori_z, _, _, _ = img.size()  # z, 1, y, x
            logits = sliding_window_inference(inputs=img, roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0, mode='gaussian')  # z, C, y', x'

            # -------------------
            # calculate loss

            loss = args.calc_loss_fun([logits], [label_GT])

            loss_sum += loss.item()

            # -------------------
            # calculate Dice

            # raw logits ---> probability distributions
            probs = F.softmax(logits, dim=1)  # z, C, y, x

            # convert to label map
            # Integer class labels
            label_pred = torch.argmax(probs, 1, keepdim=False)  # z, C, y, x ---> z, y, x

            # required by the monai
            label_pred = label_pred.unsqueeze(0).unsqueeze(0)  # z, y, x ---> 1, 1, z, y, x
            label_GT = label_GT.unsqueeze(0).unsqueeze(0)  # z, y, x ---> 1, 1, z, y, x

            dice_metric(y_pred=label_pred, y=label_GT)

            # -------------------
            # Explicit memory release
            # not useful!

            # del img
            # del logits
            # del label_pred
            # del dict_data
            # del label_GT
            # del loss
            # del probs
            
            # -------------------
            # Optional: free cached CUDA memory every few iterations
            # if (i_batch + 1) % 5 == 0:
            #     torch.cuda.empty_cache()

            torch.cuda.empty_cache()

    # -------------------
    # acc
    avg_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    # -------------------
    # loss
    avg_loss = loss_sum / (i_batch + 1)

    return avg_loss, avg_dice


# -----------------------------------------------------------------------------
# for inference

# -----------------------------------------------------------------------------
# do softmax first, and then avg
# nonlin ---> sum ---> avg 
def inference_2D_MS_Slide_Window(args, model, logger, datasetloader, save_path):

    model.eval()

    with torch.no_grad():

        for i_batch, (dict_data) in enumerate(datasetloader):
            
            img = dict_data['Img'].permute(1, 0, 2, 3).cuda()         # 1, z, y, x ---> z, 1, y, x
            label_GT = dict_data['GT'].squeeze(0).cuda()              # 1, z, y, x ---> z, y, x
            file_name = dict_data['file_name']

            # -------------------
            # zoom img for MS pred:

            ori_z, _, ori_y, ori_x = img.size()  # z, 1, y, x

            final_pred = torch.zeros([ori_z, args.num_classes, ori_y, ori_x]).cuda()

            if 1 not in args.inference_scales:
                raise ValueError('1 must be in inference_scales.')
            
            # # save for checking: z, 1, y, x ---> 1, z, y, x
            # save_img_3D(args, logger, img=img.permute(1, 0, 2, 3), file_name=file_name[0], suffix='original')

            for scale in args.inference_scales:
                
                # -------------------
                # scale, if necessary
                if scale != 1:
                    scaled_input = F.interpolate(img, scale_factor=(scale, scale), mode='area')  # z, 1, y, x ---> z, 1, y', x'

                elif scale == 1:
                    scaled_input = img

                # # save for checking: z, 3, y', x' ---> 3, z, y', x'
                # save_img_3D(args, logger, img=scaled_input.permute(1, 0, 2, 3), file_name=file_name[0], suffix='scaled_input_'+str(scale))

                # -------------------
                # sliding window inference
                if args.if_inf_flip:

                    # -------------------
                    # slide window inf
                    pred_scaled = sliding_window_inference(inputs=scaled_input, roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0.5, mode='gaussian')  # z, C, y', x'
                    
                    pred = F.softmax(pred_scaled, dim=1)  # z, C, y', x'

                    # -------------------
                    # horizontally, flip

                    pred_flipped_scaled = sliding_window_inference(inputs=torch.flip(scaled_input,  dims=[3]), roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0.5, mode='gaussian')  # z, C, y', x'

                    pred += F.softmax(torch.flip(pred_flipped_scaled, dims=[3]), dim=1) # horizontally, flip back: z, C, y', x'

                    # -------------------
                    # vertically, flip

                    pred_flipped_scaled = sliding_window_inference(inputs=torch.flip(scaled_input,  dims=[2]), roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0.5, mode='gaussian')  # z, C, y', x'

                    pred += F.softmax(torch.flip(pred_flipped_scaled, dims=[2]), dim=1) # vertically, flip back: z, C, y', x'

                    # -------------------
                    # vertically + horizontally

                    pred_flipped_scaled = sliding_window_inference(inputs=torch.flip(scaled_input,  dims=[2, 3]), roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0.5, mode='gaussian')  # z, C, y', x'

                    pred += F.softmax(torch.flip(pred_flipped_scaled, dims=[2, 3]), dim=1) # vertically, flip back: z, C, y', x'

                    # -------------------
                    # do avg
                    pred = pred / 4  # z, C, y', x'

                    logger.info("Do advanced flip for MS inference: horizontally, vertically, horizontally + vertically, slid window + gaussian, nonlin ---> sum ---> avg")

                else:

                    # slide window inf
                    pred_scaled = sliding_window_inference(inputs=scaled_input, roi_size=args.img_size, sw_batch_size=ori_z, predictor=model, overlap=0.5, mode='gaussian')  # z, C, y', x'

                    pred = F.softmax(pred_scaled, dim=1)  # z, C, y', x'

                    logger.info("Not do flip for MS inference")
                
                # -------------------
                # scale back, if necessary
                if scale != 1:
                    scaled_pred = F.interpolate(pred, size=(ori_y, ori_x), mode='area')  # z, C, y, x

                elif scale == 1:
                    scaled_pred = pred

                assert final_pred.size() == scaled_pred.size(), "must keep the same shape, after MS inference"

                # -------------------
                final_pred += scaled_pred   # z, C, y, x

            # -------------------
            pred_softmax = final_pred / len(args.inference_scales) # z, C, y, x

            # logger.info('img shape: ' + str(img.shape))
            # logger.info('pred shape: ' + str(pred.shape))
            # logger.info('pred_softmax shape: ' + str(pred_softmax.shape))
            
            # -------------------
            # save softmax predictions

            pred_softmax_np = pred_softmax.permute(1, 0, 2, 3).cpu().numpy().astype(np.float32) # z, C, y, x ---> C, z, y, x ---> cpu ---> np ---> float32

            # logger.info('softmax pred size: ' + str(pred_softmax_np.size))
            logger.info('softmax pred shape: ' + str(pred_softmax_np.shape))
            # logger.info('softmax pred ndim: ' + str(pred_softmax_np.ndim))
            # logger.info('softmax pred type: ' + str(type(pred_softmax_np)))
            # logger.info('softmax pred dtype: ' + str(pred_softmax_np.dtype))

            npz_path = os.path.join(save_path, file_name[0] + '.npz')
            np.savez(npz_path, softmax = pred_softmax_np)

            logger.info("save softmax prediction to this path: {} ".format(npz_path))
