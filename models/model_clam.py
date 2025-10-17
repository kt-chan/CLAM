import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
    ):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
    ):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [
            nn.Linear(size[1], 1) for i in range(n_classes)
        ]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A[i], h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_SB_Regression(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        min_score=3.0,
        max_score=5.0,
    ):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # Regression output for primary and secondary Gleason patterns
        self.regressors = nn.Linear(size[1], 2)  # Output: [primary, secondary]

        # Instance classifiers for pattern detection
        instance_classifiers = [
            nn.Linear(size[1], 2) for i in range(2)
        ]  # Two for primary/secondary pattern detection
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # Regression parameters
        self.min_score = min_score
        self.max_score = max_score

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # Instance-level evaluation for pattern detection
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # Instance-level evaluation for out-of-class pattern detection
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            # For regression, we use instance evaluation to detect pattern presence
            if label is not None:
                primary_label, secondary_label = extract_pattern_labels(label)

                # Create tensor from Python scalars
                inst_labels = torch.tensor(
                    [primary_label, secondary_label], device=h.device
                )

                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1:  # Pattern present
                        instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    else:  # Pattern not prominent
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, classifier
                        )

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        # Regression output for primary and secondary patterns
        pattern_logits = self.regressors(M)  # [primary, secondary]

        # Apply max/min constraint - ALWAYS differentiable
        primary_raw = pattern_logits[0, 0]
        secondary_raw = pattern_logits[0, 1]

        # Primary = max, Secondary = min (ensures primary >= secondary)
        primary_constrained = torch.max(primary_raw, secondary_raw)
        secondary_constrained = torch.min(primary_raw, secondary_raw)

        # Replace with constrained values
        pattern_logits = torch.stack(
            [primary_constrained, secondary_constrained]
        ).unsqueeze(0)

        # Apply sigmoid to get values between 0-1, then scale to min_score-max_score range
        pattern_values = (
            torch.sigmoid(pattern_logits) * (self.max_score - self.min_score)
            + self.min_score
        )

        # For final predictions - round to integers during inference
        if not self.training:
            pattern_predictions = torch.round(
                pattern_values
            )  # Gets integer scores 3,4,5
        else:
            pattern_predictions = pattern_values  # Keep continuous for gradient flow

        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({"features": M})

        return pattern_logits, pattern_values, pattern_predictions, A_raw, results_dict


class CLAM_MB_Regression(CLAM_SB_Regression):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        min_score=3.0,
        max_score=5.0,
    ):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=2
            )  # Two attention heads for primary/secondary
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=2)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # Separate regressors for primary and secondary patterns
        self.primary_regressor = nn.Linear(size[1], 1)
        self.secondary_regressor = nn.Linear(size[1], 1)

        instance_classifiers = [
            nn.Linear(size[1], 2) for i in range(2)
        ]  # For pattern detection
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # Regression parameters
        self.min_score = min_score
        self.max_score = max_score

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            if label is not None:
                primary_label, secondary_label = extract_pattern_labels(label)

                # Create tensor from Python scalars
                inst_labels = torch.tensor(
                    [primary_label, secondary_label], device=h.device
                )

                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]

                    # Use corresponding attention head for each pattern
                    pattern_attention = A[i] if i < A.shape[0] else A[0]

                    if inst_label == 1:  # Pattern present
                        instance_loss, preds, targets = self.inst_eval(
                            pattern_attention, h, classifier
                        )
                    else:  # Pattern not prominent
                        instance_loss, preds, targets = self.inst_eval_out(
                            pattern_attention, h, classifier
                        )

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    total_inst_loss += instance_loss

                # Average the loss
                total_inst_loss /= len(self.instance_classifiers)

        # Multi-branch: use separate attention for primary and secondary
        M_primary = torch.mm(A[0:1], h)  # Use first attention head for primary
        M_secondary = torch.mm(A[1:2], h)  # Use second attention head for secondary

        # Get predictions
        primary_logit = self.primary_regressor(M_primary)
        secondary_logit = self.secondary_regressor(M_secondary)

        # Combine into final output
        pattern_logits = torch.cat([primary_logit, secondary_logit], dim=1)

        # Apply max/min constraint - ALWAYS differentiable
        primary_raw = pattern_logits[0, 0]
        secondary_raw = pattern_logits[0, 1]

        ### The system is based on two main components: the Primary Score and the Secondary Score, each of which is a Gleason Grade with a value range from 3 to 5. A pathologist examines tissue samples from a prostate biopsy and assigns a grade from 1 to 5 to the two most common patterns of cancer cells observed. Higher grades indicate that the cancer cells look more abnormal or "undifferentiated," suggesting a more aggressive tumor.
        # The Primary Score is the Gleason Grade assigned to the most common (predominant) pattern of cancer cells found in the tissue sample. And the Secondary Score is the Gleason Grade assigned to the second most common pattern of cancer cells found in the tissue sample.

        ### Uncommnet below code if you assume the Primary Score is always larger than the secondary score
        ### this is not true according to feedback from hospital expert.
        # primary_raw = torch.max(primary_raw, secondary_raw)
        # secondary_raw = torch.min(primary_raw, secondary_raw)

        # Replace with constrained values
        pattern_logits = torch.stack([primary_raw, secondary_raw]).unsqueeze(0)

        # Scale to Gleason score range
        pattern_values = (
            torch.sigmoid(pattern_logits) * (self.max_score - self.min_score)
            + self.min_score
        )

        # Round to integers during inference
        if not self.training:
            pattern_predictions = torch.round(pattern_values)
        else:
            pattern_predictions = pattern_values

        if instance_eval:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}
        if return_features:
            results_dict.update(
                {"features_primary": M_primary, "features_secondary": M_secondary}
            )

        return pattern_logits, pattern_values, pattern_predictions, A_raw, results_dict


def extract_pattern_labels(label):
    """Extract primary and secondary pattern labels from various label formats"""
    if label is None:
        return None, None

    # Handle different label formats
    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            # Single value - assume both patterns same
            primary_value = label.item()
            secondary_value = primary_value
        else:
            # Multiple values - handle different tensor shapes
            label_flat = label.flatten()
            if len(label_flat) >= 2:
                primary_value = label_flat[0].item()
                secondary_value = label_flat[1].item()
            else:
                primary_value = label_flat[0].item()
                secondary_value = primary_value
    elif isinstance(label, (list, tuple)):
        # List or tuple format
        primary_value = float(label[0])
        secondary_value = float(label[1]) if len(label) > 1 else primary_value
    else:
        # Single scalar value
        primary_value = float(label)
        secondary_value = primary_value

    # Convert to binary labels for pattern detection (≥3 = clinically significant)
    primary_label = 1 if primary_value >= 3 else 0
    secondary_label = 1 if secondary_value >= 3 else 0

    return primary_label, secondary_label
