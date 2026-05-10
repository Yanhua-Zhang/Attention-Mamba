from torch import nn


# -----------------------------------------------------------------------------
#   use this if you have several outputs and ground truth (both list of same len) and the loss should be computed between them (inputs[0] and labels[0], inputs[1] and labels[1] etc)

class calc_MultipleOutput_Loss(nn.Module):
    def __init__(self, loss, loss_weights=None):

        super(calc_MultipleOutput_Loss, self).__init__()
        self.loss_weights = loss_weights
        self.loss = loss

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        # train: 2D input: b, N, y, x; 2D label: b, 1, y, x
        # val: 2D input: b, N, y, x; 2D label: b, y, x
        all_loss = weights[0] * self.loss(inputs[0], label_list[0])

        # aux loss
        for i in range(1, len(inputs)):
            assert len(inputs) == len(weights), "inputs has the same length as weights"
            all_loss += weights[i] * self.loss(inputs[i], label_list[0])

        return all_loss
    

# -----------------------------------------------------------------------------
# for using CE loss

class calc_MultipleOutput_CE_Loss(nn.Module):
    def __init__(self, loss, loss_weights=None):

        super(calc_MultipleOutput_CE_Loss, self).__init__()
        self.loss_weights = loss_weights
        self.loss = loss

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        # for CE_and_Dice_loss, label: b, 1, h, w ---> b, h, w
        all_loss = weights[0] * self.loss(inputs[0], label_list[0].squeeze(1))

        # aux loss
        for i in range(1, len(inputs)):
            assert len(inputs) == len(weights), "inputs has the same length as weights"

            # for CE_and_Dice_loss, label: b, 1, h, w ---> b, h, w
            all_loss += weights[i] * self.loss(inputs[i], label_list[0].squeeze(1))

        return all_loss


# used for the balanced CE loss: CE_and_Dice_loss_V2
class calc_MultipleOutput_CE_Loss_V2(nn.Module):
    def __init__(self, loss, loss_weights=None):

        super(calc_MultipleOutput_CE_Loss_V2, self).__init__()
        self.loss_weights = loss_weights
        self.loss = loss

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        all_loss = weights[0] * self.loss(inputs[0], label_list[0])

        # aux loss
        for i in range(1, len(inputs)):
            assert len(inputs) == len(weights), "inputs has the same length as weights"

            all_loss += weights[i] * self.loss(inputs[i], label_list[0])

        return all_loss


# -----------------------------------------------------------------------------

class calc_MultiOutput_Align_Loss(nn.Module):
    def __init__(self, args, loss_lesion, loss_prostate):

        super(calc_MultiOutput_Align_Loss, self).__init__()

        self.loss_lesion = loss_lesion
        self.loss_prostate = loss_prostate
        self.args = args

        self.loss_weights = args.loss_weights

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        all_loss = weights[0] * self.loss_lesion(inputs[0], label_list[0])

        if len(inputs) > 1:
            # aux loss
            for i in range(1, len(self.args.aux_heads_choose) + 1): 
                assert len(inputs) == len(weights), "inputs has the same length as weights"
                all_loss += weights[i] * self.loss_lesion(inputs[i], label_list[0])

            if self.args.If_Deep_Fusion_Supervision:
                # align/fuse loss
                for i in range(len(self.args.aux_heads_choose) + 1, len(inputs)):  
                    assert len(inputs) == len(weights), "inputs has the same length as weights"
                    # for CE_and_Dice_loss, label_prostate: b, 1, h, w ---> b, h, w
                    all_loss += weights[i] * self.loss_prostate(inputs[i], label_list[1].squeeze(1))

        return all_loss


# -----------------------------------------------------------------------------
# add Multi-scale fus loss

class calc_MultiOutput_Align_Multi_Scale_Fus_Loss(nn.Module):
    def __init__(self, args, loss_lesion, loss_prostate):

        super(calc_MultiOutput_Align_Multi_Scale_Fus_Loss, self).__init__()

        self.loss_lesion = loss_lesion
        self.loss_prostate = loss_prostate
        self.args = args

        self.loss_weights = args.loss_weights

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        all_loss = weights[0] * self.loss_lesion(inputs[0], label_list[0])

        if len(inputs) > 1:
            # aux loss
            for i in range(1, len(self.args.aux_heads_choose) + 1): 
                assert len(inputs) == len(weights), "inputs has the same length as weights"
                # lesion loss
                all_loss += weights[i] * self.loss_lesion(inputs[i], label_list[0])

            if self.args.If_Deep_Fusion_Supervision:
                # align/fuse loss
                for i in range(len(self.args.aux_heads_choose) + 1, len(self.args.aux_heads_choose) + 1 + 4):  
                    assert len(inputs) == len(self.args.aux_heads_choose) + 6, "inputs has the same length as weights"
                    # for CE_and_Dice_loss, label_prostate: b, 1, h, w ---> b, h, w
                    all_loss += weights[i] * self.loss_prostate(inputs[i], label_list[1].squeeze(1))

            if self.args.If_Multi_scale_Fusion_Supervision:
                # Multi-scale fusion Sup
                for i in range(len(self.args.aux_heads_choose) + 1 + 4, len(inputs)):  
                    assert len(inputs) == len(self.args.aux_heads_choose) + 6, "inputs has the same length as weights"
                    # lesion loss
                    all_loss += weights[i] * self.loss_lesion(inputs[i], label_list[0])

        return all_loss


# -----------------------------------------------------------------------------
from loss_functions.loss import Inconsistency_loss

class calc_MultiOutput_Inconsistenc_Loss(nn.Module):
    def __init__(self, args, loss_lesion, loss_prostate):

        super(calc_MultiOutput_Inconsistenc_Loss, self).__init__()
        
        self.loss_lesion = loss_lesion
        self.loss_prostate = loss_prostate
        self.args = args

        self.if_add_inconsist_loss = args.if_add_inconsist_loss
        self.loss_weights = args.loss_weights
        self.weight_inconsist = args.weight_inconsist
        self.inconsist_loss_fun = Inconsistency_loss(nonlin_name=args.nonlin_name, loss_type=args.inconsist_loss_type)

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # -------------------
        # main loss
        all_loss = weights[0] * self.loss_lesion(inputs[0], label_list[0])

        # -------------------
        if len(inputs) > 1:

            assert len(inputs) == (len(self.args.aux_heads_choose) + 4), "inputs has specific length"
            assert (len(inputs) + 1) == len(weights), "inputs has the same length as weights"

            # -------------------
            # aux loss
            for i in range(1, len(self.args.aux_heads_choose) + 1): 
                
                all_loss += weights[i] * self.loss_lesion(inputs[i], label_list[0])

            # -------------------
            # prostate loss + align/fuse/inconsistency loss
            list_dwi = inputs[-1]
            list_adc = inputs[-2]
            list_t2 = inputs[-3]

            assert isinstance(list_dwi, (list)), "must be list"
            assert isinstance(list_adc, (list)), "must be list"
            assert isinstance(list_t2, (list)), "must be list"

            for i in range(4):

                pre_dwi = list_dwi[i]
                pre_adc = list_adc[i]
                pre_t2 = list_t2[i]

                # prostate loss
                all_loss += weights[i] * (self.loss_prostate(pre_t2, label_list[1].squeeze(1)) + self.loss_prostate(pre_dwi, label_list[1].squeeze(1)) + self.loss_prostate(pre_adc, label_list[1].squeeze(1))) 

                if self.if_add_inconsist_loss:
                    # inconsistency loss                          
                    all_loss += self.weight_inconsist * (self.inconsist_loss_fun(pre_t2, pre_dwi) + self.inconsist_loss_fun(pre_t2, pre_adc))
            
        return all_loss
    

# -----------------------------------------------------------------------------

class calc_MultiOutput_Align_Fea_Inconsist_Loss(nn.Module):
    def __init__(self, args, loss_lesion, loss_prostate):

        super(calc_MultiOutput_Align_Fea_Inconsist_Loss, self).__init__()

        self.loss_lesion = loss_lesion
        self.loss_prostate = loss_prostate
        self.args = args

        self.if_add_inconsist_loss = args.if_add_inconsist_loss
        self.loss_weights = args.loss_weights
        self.weight_inconsist = args.weight_inconsist
        self.inconsist_loss_fun = Inconsistency_loss(nonlin_name=args.nonlin_name, loss_type=args.inconsist_loss_type)

    def forward(self, inputs, label_list):

        # label_list: [label_lesion, label_prostate, ....]

        if self.loss_weights is None:
            weights = [1] * len(inputs)
        else:
            weights = self.loss_weights

        assert isinstance(inputs, (tuple, list)), "inputs must be either tuple or list"
        assert isinstance(label_list, (tuple, list)), "labels must be either tuple or list"
        assert isinstance(weights, (list)), "weights must be list"
        # assert len(inputs) == len(labels), "inputs has the same length as labels"

        # main loss
        all_loss = weights[0] * self.loss_lesion(inputs[0], label_list[0])

        if len(inputs) > 1:

            assert len(inputs) == (len(self.args.aux_heads_choose) + 8), "inputs has specific length"
            assert (len(inputs) - 3) == len(weights), "inputs has the same length as weights"

            # -------------------
            # aux loss
            for i in range(1, len(self.args.aux_heads_choose) + 1): 
                all_loss += weights[i] * self.loss_lesion(inputs[i], label_list[0])

            # -------------------
            # align/fuse loss
            for i in range(len(self.args.aux_heads_choose) + 1, len(inputs) - 3):  
                # for CE_and_Dice_loss, label_prostate: b, 1, h, w ---> b, h, w
                all_loss += weights[i] * self.loss_prostate(inputs[i], label_list[1].squeeze(1))

            # -------------------
            # feature inconsistency loss

            if self.if_add_inconsist_loss:

                list_dwi = inputs[-1]
                list_adc = inputs[-2]
                list_t2 = inputs[-3]

                assert isinstance(list_dwi, (list)), "must be list"
                assert isinstance(list_adc, (list)), "must be list"
                assert isinstance(list_t2, (list)), "must be list"

                for i in range(4):

                    fea_dwi = list_dwi[i]
                    fea_adc = list_adc[i]
                    fea_t2 = list_t2[i]
        
                    # inconsistency loss                          
                    all_loss += self.weight_inconsist * (self.inconsist_loss_fun(fea_t2, fea_dwi) + self.inconsist_loss_fun(fea_t2, fea_adc))

            else:
                
                raise ValueError('the inconsist loss much be added for this loss calculator.')


        return all_loss