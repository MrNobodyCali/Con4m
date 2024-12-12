import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_layer import BertModel


class ChannelNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x.size(): BatchSize * ChannelNum * SeqLen
        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean) * torch.rsqrt(cum_var + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CLEncoder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(CLEncoder, self).__init__()
        # CNN definition
        norm_layer = ChannelNorm
        cnn_layer_list = []

        in_channels_list = [config.n_features] + [config.d_model for _ in range(len(config.kernel_size_list) - 1)]
        out_channels_list = [config.d_model for _ in range(len(config.kernel_size_list))]
        for i in range(len(config.kernel_size_list)):
            cnn_layer_list.append(
                nn.Conv1d(
                    in_channels=in_channels_list[i],
                    out_channels=out_channels_list[i],
                    kernel_size=config.kernel_size_list[i],
                    stride=config.stride_size_list[i],
                    padding=config.padding_size_list[i],
                ))
            cnn_layer_list.append(norm_layer(config.d_model))
            cnn_layer_list.append(nn.ReLU())

        self.cnn_layers = nn.Sequential(*cnn_layer_list)
        self.down_sampling = config.down_sampling

        # Transformer definition
        local_config = copy.deepcopy(config)
        local_config.seg_small_num = config.raw_input_len // config.down_sampling
        self.transformer = BertModel(local_config)

    def forward(self, x):
        # x.size(): batch_size x n_features x length
        local_r = self.cnn_layers(x).permute(0, 2, 1)
        # local_r.size(): batch_size x cnn_length x d_model
        context_r = self.transformer(local_r)
        # context_r.size(): batch_size x cnn_length x d_model
        seg_r = context_r.mean(dim=1)
        # seg_r.size(): batch_size x d_model
        return seg_r


class CLPredictor(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(CLPredictor, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_inner),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inner, config.d_inner // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inner // 2, config.n_class),
        )
        self.class_criterion = nn.CrossEntropyLoss()

        self.same_classifier = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_inner),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inner, config.d_inner // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inner // 2, 2),
        )
        self.same_criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            agg_r,
            y,
    ):
        # y.size(): batch_size x seg_small_num x n_class ->
        # seg_y.size(): batch_size x seg_small_num
        seg_y = torch.argmax(y, dim=-1)

        # agg_r.size(): batch_size x seg_small_num x d_model
        # Predict the independent labels
        logit_p = self.class_classifier(agg_r)
        # hat_p.size(): batch_size x q_len x n_class
        hat_p = F.softmax(logit_p, dim=-1).detach()
        loss1 = self.class_criterion(logit_p.view(-1, logit_p.size(-1)), seg_y.view(-1))

        # Predict the neighbor's labels
        # logit_q.size(): batch_size x q_len x k_len x 2
        logit_q = self.same_classifier(
            torch.cat(torch.broadcast_tensors(agg_r[:, :, None, :], agg_r[:, None, :, :]), dim=-1)
        )
        # hat_q_1.size(): batch_size x q_len x k_len x 1
        hat_q_1 = F.softmax(logit_q, dim=-1)[:, :, :, [1]].detach()
        same_y = (y[:, :, None, :] * y[:, None, :, :]).max(dim=-1)[0]
        loss2 = self.same_criterion(logit_q.view(-1, logit_q.size(-1)), same_y.view(-1))

        # Loss
        loss = loss1 + loss2

        # Two types of labels
        # tilde_p.size(): batch_size x q_len x k_len x n_class
        #              -> batch_size x q_len x n_class
        tilde_p = (hat_q_1 * hat_p[:, None, :, :]).sum(dim=-2)
        # Rescale
        tilde_p = tilde_p / tilde_p.sum(dim=-1, keepdim=True)

        return loss, hat_p, tilde_p, seg_y


class CLModel(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(CLModel, self).__init__()
        self.encoder = CLEncoder(config)
        self.aggregator = BertModel(config)
        self.predictor = CLPredictor(config)

    def forward(
            self,
            x,
            y,
    ):
        # x.size(): batch_size x seg_small_num x n_features x length
        # y.size(): batch_size x seg_small_num x n_class

        # Step 1: encode the small segments
        seg_r = self.encoder(x.view(-1, *x.size()[-2:]))
        # seg_r.size(): (batch_size * seg_small_num) x d_model ->
        #                batch_size x seg_small_num x d_model
        seg_r = seg_r.view(y.size()[:2] + seg_r.size()[1:])

        # Step 2: obtain the continuous representations
        # agg_r.size(): batch_size x seg_small_num x d_model
        agg_r = self.aggregator(seg_r)

        # Step 3: obtain the coherent predictions
        loss, hat_p, tilde_p, seg_y = self.predictor(
            agg_r,
            y,
        )

        return loss, hat_p, tilde_p, seg_y
