import warnings
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
import copy


class Alpha(nn.Module):
    def __init__(self, num_classes, alpha):
        super(Alpha, self).__init__()
        self.weight = nn.Parameter(alpha * torch.ones(num_classes, 1).half())

    def forward(self, x):
        return x * self.weight.t()

    def __str__(self) -> str:
        return self.weight


class ClipAdapter(nn.Module):
    def __init__(
        self,
        model,
        dataloader=None,
        classnames=None,
        alpha=5,
        beta=1,
        augment_epoch=10,
        device="cuda:0",
        manual_cache=False,
    ) -> None:
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.augment_epoch = augment_epoch
        self.classnames = classnames
        self.device = device
        self.text_features = self._encoder_text().to(self.device)
        if not manual_cache and dataloader is not None:
            self.cache_keys, self.cache_values = self._bulid_cache(
                dataloader, self.augment_epoch
            )

    def reset_text(self, texts):
        self.classnames = texts
        self.text_features = self._encoder_text()

    def _encoder_text(self):
        with torch.no_grad():
            text = clip.tokenize(self.classnames)
            text_features = self.model.encode_text(text.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def _bulid_cache(self, dataloader, augment_epoch):
        cache_keys = []
        cache_values = []

        for augment_idx in tqdm(range(augment_epoch)):
            train_features = []
            for i, (images, target, _) in enumerate(dataloader):
                image_features = self.model.encode_image(images.to(self.device))
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).to(self.device)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half().to(self.device)

        return cache_keys, cache_values

    def add_cache(self, img_feature, label):
        """
        img_feature: output from Clip encode_image, need normlization in dim=-1, shape: batch_size * feature_dim
        """
        assert img_feature.shape[0] == label.shape[0]
        one_hot = (
            F.one_hot(label, num_classes=len(self.classnames)).half().to(self.device)
        )
        if not hasattr(self, "cache_keys"):
            self.cache_keys = img_feature.t()
            self.cache_values = one_hot
        else:
            self.cache_keys = torch.cat([self.cache_keys, img_feature.t()], dim=1)
            self.cache_values = torch.cat([self.cache_values, one_hot], dim=0)

    def cal_entropy(self):
        self.entropy = F.cross_entropy(
            100 * self.cache_keys.t() @ self.text_features.t(),
            self.cache_values.argmax(dim=-1),
            reduction="none",
        )
        return self.entropy

    def clear_cache(self):
        if hasattr(self, "cache_keys"):
            del self.cache_keys
            del self.cache_values
            return True
        else:
            return False

    def rebulid_cache(self, dataloader, augment_epoch=None):
        if augment_epoch is None:
            augment_epoch = self.augment_epoch
        self.cache_keys, self.cache_values = self._bulid_cache(
            dataloader=dataloader, augment_epoch=augment_epoch
        )

    def pre_load_features(self, dataloader, store=True):
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target, _) in enumerate(tqdm(dataloader)):
                images, target = images.to(self.device), target.to(self.device)
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        if store:
            self.test_features = features
            self.test_labels = labels
            self.test_clip_logits = 100.0 * features @ self.text_features.t()
        return features, labels

    def eval(self, dataloader=None, adapt=True, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if dataloader is not None:
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for i, (imgs, label, _) in enumerate(tqdm(dataloader)):
                    logits = self.__call__(imgs.to(self.device), adapt=adapt)
                    probs = logits.softmax(dim=-1)
                    pred_label = probs.argmax(dim=1)
                    all_targets.extend(label.cpu().numpy())
                    all_predictions.extend(pred_label.cpu().numpy())
        elif hasattr(self, "test_features") and hasattr(self, "test_labels"):
            all_targets = self.test_labels.cpu().numpy()
            if adapt:
                fused_logits = self._fuse_logits(
                    self.test_features,
                    self.test_clip_logits,
                    alpha=alpha,
                    beta=beta,
                )
                all_predictions = (
                    fused_logits.softmax(dim=-1).argmax(dim=1).cpu().numpy()
                )
            else:
                all_predictions = (
                    self.test_clip_logits.softmax(dim=-1).argmax(dim=1).cpu().numpy()
                )
        accuracy = metrics.accuracy_score(all_targets, all_predictions)
        precision = metrics.precision_score(all_targets, all_predictions, average=None)
        recall = metrics.recall_score(all_targets, all_predictions, average=None)
        f1 = metrics.f1_score(all_targets, all_predictions, average=None)
        return all_predictions, all_targets, (accuracy, precision, recall, f1)

    def train_keys(
        self,
        dataloader,
        epoch=10,
        dataloader_eval=None,
        search_hp=True,
        alpha_train=False,
        beta_train=False,
    ):
        if dataloader_eval is not None:
            test_features, test_labels = self.pre_load_features(
                dataloader=dataloader_eval
            )
        elif hasattr(self, "test_features") and hasattr(self, "test_labels"):
            test_features, test_labels = self.test_features, self.test_labels

        # Enable the cached keys to be learnable
        adapter = nn.Linear(
            self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False
        ).to(self.device)
        adapter.weight = nn.Parameter(self.cache_keys.t())
        if alpha_train:
            self.alpha_weight = Alpha(len(self.classnames), self.alpha).to(self.device)
            lr_list = [
                {"params": self.alpha_weight.parameters(), "lr": 0.05},
                {"params": adapter.parameters(), "lr": 0.001},
            ]
            if beta_train:
                self.beta_matrix = nn.Linear(
                    self.cache_keys.shape[1], self.cache_keys.shape[1], bias=False
                ).to(self.device)
                self.beta_matrix.weight = nn.Parameter(
                    torch.diag(
                        torch.tensor([self.beta] * self.cache_keys.shape[1]).half()
                    ).to(self.device)
                )
                lr_list.append({"params": self.beta_matrix.parameters(), "lr": 0.001})

            # for param in adapter.parameters():
            #     param.requires_grad = False
            optimizer = torch.optim.AdamW(
                lr_list,
                eps=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epoch * len(dataloader), gamma=0.8
            )
        else:
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, eps=1e-4)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, epoch * len(dataloader), eta_min=0.01 * 0.1
            # )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, min(epoch * len(dataloader) / 2,10 * len(dataloader)), gamma=0.8, last_epoch=-1
            )

        best_acc, best_epoch = 0.0, 0
        for train_idx in range(epoch):
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print("Train Epoch: {:} / {:}".format(train_idx, epoch))

            for i, (images, target, _) in enumerate(tqdm(dataloader)):
                images, target = images.to(self.device), target.to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_logits = 100.0 * image_features @ self.text_features.t()

                fused_logits = self._fuse_logits(
                    image_featrues=image_features,
                    clip_logits=clip_logits,
                    adapter=adapter,
                )
                loss = F.cross_entropy(fused_logits, target)
                #
                probs = fused_logits.softmax(dim=-1)
                pred_label = probs.argmax(dim=1)
                accuracy = metrics.accuracy_score(
                    target.cpu().numpy(), pred_label.cpu().numpy()
                )
                correct_samples += accuracy * len(fused_logits)
                all_samples += len(fused_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(
                "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                    current_lr,
                    correct_samples / all_samples,
                    correct_samples,
                    all_samples,
                    sum(loss_list) / len(loss_list),
                )
            )
            if search_hp:
                self.search_hp(alpha_train=alpha_train)
            adapter.eval()
            clip_logits = 100.0 * test_features @ self.text_features.t()
            fused_logits = self._fuse_logits(
                image_featrues=test_features, clip_logits=clip_logits, adapter=adapter
            )
            probs = fused_logits.softmax(dim=-1)
            pred_label = probs.argmax(dim=1)
            accuracy = metrics.accuracy_score(
                test_labels.cpu().numpy(), pred_label.cpu().numpy()
            )
            print(
                "**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(
                    accuracy * 100
                )
            )
            if hasattr(self, "beta_matrix"):
                print("Beta:{}\n".format(self.beta_matrix.weight))
            if hasattr(self, "alpha_weight"):
                print("Alpha:{}\n".format(self.alpha_weight.weight))

            if accuracy > best_acc:
                best_adapter_weight = copy.deepcopy(adapter.state_dict())
                if alpha_train:
                    best_alpha_weight = copy.deepcopy(self.alpha_weight.state_dict())
                best_acc = accuracy
                best_epoch = train_idx
        print(
            f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc*100:.2f}, at epoch: {best_epoch}. ****\n"
        )
        adapter.load_state_dict(best_adapter_weight)
        if alpha_train:
            self.alpha_weight.load_state_dict(best_alpha_weight)

    def save(self, filepath):
        params = {
            "alpha": self.alpha,
            "beta": self.beta,
        }
        if hasattr(self, "cache_keys"):
            params["cache_keys"] = self.cache_keys.cpu()
        if hasattr(self, "cache_values"):
            params["cache_values"] = self.cache_values.cpu()
        if hasattr(self, "alpha_weight"):
            params["alpha_weight"] = self.alpha_weight.cpu()
        if hasattr(self, "beta_matrix"):
            params["beta_matrix"] = self.beta_matrix.cpu()
        torch.save(params, filepath)

    def load(self, filepath):
        params = torch.load(filepath)
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        if "cache_keys" in params.keys():
            self.cache_keys = params["cache_keys"].to(self.device)
        if "cache_values" in params.keys():
            self.cache_values = params["cache_values"].to(self.device)
        if "alpha_weight" in params.keys():
            self.alpha_weight = params["alpha_weight"].to(self.device)
        if "beta_matrix" in params.keys():
            self.beta_matrix = params["beta_matrix"].to(self.device)

    def search_hp(
        self,
        dataloader=None,
        search_scale=[50, 50],
        search_step=[200, 20],
        inplace=True,
        beta_search=False,
        alpha_train=False,
    ):
        if dataloader is not None:
            features, labels = self.pre_load_features(dataloader=dataloader)
        elif hasattr(self, "test_features") and hasattr(self, "test_labels"):
            features, labels = self.test_features, self.test_labels
        beta_list = [1]
        if beta_search:
            beta_list = [
                i * (search_scale[0] - 0.1) / search_step[0] + 0.1
                for i in range(search_step[0])
            ]
        alpha_list = [
            i * (search_scale[1] - 0.1) / search_step[1] + 0.1
            for i in range(search_step[1])
        ]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        affinity = features @ self.cache_keys
        clip_logits = 100.0 * features @ self.text_features.t()
        for beta in beta_list:
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
            if alpha_train:
                fused_logits = clip_logits + self.alpha_weight(cache_logits)
                probs = fused_logits.softmax(dim=-1)
                pred_label = probs.argmax(dim=1)
                accuracy = metrics.accuracy_score(
                    labels.cpu().numpy(), pred_label.cpu().numpy()
                )

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_beta = beta
                    best_alpha = self.alpha_weight.weight

            else:
                for alpha in alpha_list:
                    fused_logits = clip_logits + cache_logits * alpha
                    probs = fused_logits.softmax(dim=-1)
                    pred_label = probs.argmax(dim=1)
                    accuracy = metrics.accuracy_score(
                        labels.cpu().numpy(), pred_label.cpu().numpy()
                    )

                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_beta = beta
                        best_alpha = alpha

        if alpha_train:
            print(
                "New best HP, beta: {:.2f}, alpha: {}; accuracy: {:.2f}".format(
                    best_beta, best_alpha, best_acc * 100
                )
            )
        else:
            print(
                "New best HP, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(
                    best_beta, best_alpha, best_acc * 100
                )
            )

        if inplace:
            self.alpha = best_alpha
            self.beta = best_beta
        return best_beta, best_alpha, best_acc * 100

    def _fuse_logits(
        self, image_featrues, clip_logits, adapter=None, alpha=None, beta=None
    ):
        if alpha is None and not hasattr(self, "alpha_weight"):
            alpha = self.alpha
        if beta is None and not hasattr(self, "beta_matrix"):
            beta = self.beta

        if hasattr(self, "cache_keys") and hasattr(self, "cache_values"):
            if adapter is not None:
                affinity = adapter(image_featrues)
            else:
                affinity = image_featrues @ self.cache_keys

            if hasattr(self, "beta_matrix"):
                sharpness = self.beta_matrix(1 - affinity)
            else:
                sharpness = beta * (1 - affinity)
            cache_logits = ((-1) * sharpness).exp() @ self.cache_values

            if hasattr(self, "alpha_weight"):
                tip_logits = clip_logits + self.alpha_weight(cache_logits)
            else:
                tip_logits = clip_logits + cache_logits * alpha

            return tip_logits
        else:
            warnings.warn("No Cached Error, Turn to Plain Mode")
            return clip_logits

    @torch.no_grad()
    def __call__(self, images, adapt=True, alpha=None, beta=None):
        images = images.to(self.device)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        clip_logits = 100.0 * image_features @ self.text_features.t()
        if adapt:
            fused_logits = self._fuse_logits(
                image_featrues=image_features,
                clip_logits=clip_logits,
                alpha=alpha,
                beta=beta,
            )
            return fused_logits.softmax(dim=-1)
        return clip_logits.softmax(dim=-1)
