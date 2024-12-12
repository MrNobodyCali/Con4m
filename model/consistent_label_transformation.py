import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTanh(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LearnableTanh, self).__init__()
        self.a = nn.parameter.Parameter(torch.ones(batch_size, 1))
        self.b = nn.parameter.Parameter(torch.ones(batch_size, 1))
        self.c = nn.parameter.Parameter(torch.ones(batch_size, 1))
        self.d = nn.parameter.Parameter(torch.ones(batch_size, 1))
        self.seq_len = seq_len
        self.register_buffer("x", torch.arange(seq_len).expand((1, -1)) - seq_len // 2)

    def obtain_fit(self, y):
        ss = y.size(0)
        fit = self.a[:ss] * torch.tanh(self.b[:ss] * (self.x + self.c[:ss])) + self.d[:ss]
        fit_d = self.d[:ss]
        return fit, fit_d

    def forward(self, y):
        # y.size(): batch_size x seg_small_num
        ss = y.size(0)
        fit = self.a[:ss] * torch.tanh(self.b[:ss] * (self.x + self.c[:ss])) + self.d[:ss]
        loss = (fit - y) ** 2
        return loss.sum()


class CLTransform:
    def __init__(
            self,
            ori_y,
            batch_size,
            seg_num,
    ):
        # For updating labels
        # One-hot labels
        # ori_y.size(): n_level x seg_big_num x seg_small_num x n_class
        self.ori_y = ori_y
        self.his_y = self.ori_y.clone()
        # Labels of historic epochs
        # n_level x seg_big_num x seg_small_num x n_class x 4
        self.his_sin_p = torch.stack([self.ori_y.clone() for _ in range(4)], dim=-1).float()    # independent prediction
        self.his_con_p = torch.stack([self.ori_y.clone() for _ in range(4)], dim=-1).float()    # contextual prediction
        self.change_label_num = 0

        # For learning consistent labels
        self.batch_size = batch_size
        self.seg_num = seg_num
        self.l_tanh = LearnableTanh(batch_size, seg_num)
        self.lr = 0.1
        self.optimizer = torch.optim.Adam(self.l_tanh.parameters(), lr=self.lr)
        self.fit_epoch = 100
        self.tolerance = 1e-6

    # Return labels of the new epoch
    def get_correct_label(self):
        print(f'Total number of change labels: {self.change_label_num}')
        self.change_label_num = 0
        return self.his_y

    def __fit_consistent_label__(self, prob):
        # prob.size(): batch_size x seg_small_num x n_class
        batch_size, _, n_class = prob.size()
        # Rescale to (-1, 1)
        prob = 2 * prob - 1

        def __fit_tanh__(p_c):
            # Designed initialization
            diff = torch.diff(p_c, dim=-1)
            b_, c_ = torch.abs(diff).max(dim=-1, keepdim=True)
            batch_id_ = torch.arange(batch_size, dtype=c_.dtype, device=c_.device)
            b_ = b_ * torch.sign(diff[batch_id_, c_.squeeze(dim=-1)])[:, None]
            c_ = -(c_ + 0.5 - self.l_tanh.seq_len // 2)

            with torch.no_grad():
                batch_gap = self.batch_size - batch_size
                if batch_gap > 0:
                    fill_value = torch.ones(batch_gap, 1, dtype=b_.dtype, device=b_.device)
                    b_ = torch.cat([b_, fill_value], dim=0)
                    c_ = torch.cat([c_, fill_value], dim=0)
                for name, param in self.l_tanh.named_parameters():
                    if 'b' in name:
                        param.copy_(b_)
                    if 'c' in name:
                        param.copy_(c_)

            last_loss = 0.
            for _ in range(self.fit_epoch):
                loss = self.l_tanh(p_c)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if torch.abs(last_loss - loss.detach()) < self.tolerance:
                    break
                last_loss = loss.detach()

            with torch.no_grad():
                return self.l_tanh.obtain_fit(p_c)

        fit_for_class = []
        fit_mask_matrix = []
        for c in range(n_class):
            fit, fit_d = __fit_tanh__(prob[:, :, c])
            fit_for_class.append(fit)
            fit_mask_matrix.append(fit >= fit_d)

        # fit_*.size(): batch_size x seg_small_num x n_class
        fit_for_class = (torch.stack(fit_for_class, dim=-1) + 1) / 2
        fit_for_class = fit_for_class / fit_for_class.sum(dim=-1, keepdim=True)
        return fit_for_class

    def process_batch_label(
            self,
            hat_p,
            tilde_p,
            index,
            eta,
            update_label,
            epoch_num,
    ):
        # Compute the correct label
        # eta == 1 for valid and test stages
        con_p = self.__fit_consistent_label__(tilde_p)

        if update_label:
            eta = eta.unsqueeze(dim=-1)
            # bsz x seg_len x n_class
            ori_label = (self.ori_y[index[:, 0], index[:, 1]]).to(hat_p.device)

            # bsz x seg_len x n_class x 5
            his_sin_logit = torch.cat([hat_p.unsqueeze(dim=-1),
                                       self.his_sin_p[index[:, 0], index[:, 1]].to(hat_p.device)], dim=-1)
            self.his_sin_p[index[:, 0], index[:, 1]] = his_sin_logit[:, :, :, :4].cpu()
            # bsz x seg_len x n_class x 5
            his_con_logit = torch.cat([con_p.unsqueeze(dim=-1),
                                       self.his_con_p[index[:, 0], index[:, 1]].to(hat_p.device)], dim=-1)
            self.his_con_p[index[:, 0], index[:, 1]] = his_con_logit[:, :, :, :4].cpu()

            # w_e
            weight_factor = torch.exp((epoch_num - torch.arange(5)) / 2.).to(hat_p.device)
            weight_factor = (weight_factor / weight_factor.sum(dim=-1))[None, None, None, :]

            # bsz x seg_len x n_class x 5 -> bsz x seg_len x n_class
            his_sin_logit = (his_sin_logit * weight_factor).sum(dim=-1)
            his_con_logit = (his_con_logit * weight_factor).sum(dim=-1)

            mix_prob = (1 - eta) * ori_label + eta * ((1 - eta / 2) * his_sin_logit + (eta / 2) * his_con_logit)
        # For warm-up and valid/test stage
        else:
            mix_prob = 0.5 * hat_p + 0.5 * con_p

        mix_prob = mix_prob / mix_prob.sum(dim=-1, keepdim=True)
        consistent_label = torch.argmax(mix_prob, dim=-1)
        if update_label:
            self.change_label_num += torch.argmax(self.his_y[index[:, 0], index[:, 1]], dim=-1)\
                .ne(consistent_label.cpu()).sum()
            self.his_y[index[:, 0], index[:, 1]] = F.one_hot(consistent_label, num_classes=mix_prob.size(-1)).cpu()

        return consistent_label
