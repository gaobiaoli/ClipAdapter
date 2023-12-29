import warnings
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics


class ClipAdapter:
    def __init__(
        self,
        model,
        dataloader=None,
        classnames=None,
        alpha=20,
        beta=20,
        augment_epoch=10,
        device="cuda:0",
        manual_cache=False,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.cfg = {"alpha": alpha, "beta": beta, "augment_epoch": augment_epoch}

        self.classnames = classnames
        self.device = device

        self.text_features = self._encoder_text()
        if not manual_cache:
            self.cache_keys, self.cache_values = self._bulid_cache(
                dataloader, self.cfg["augment_epoch"]
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

    def _bulid_cache(self, dataloader, augment_epoch):
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(augment_epoch):
                train_features = []

                print("Augment Epoch: {:} / {:}".format(augment_idx, augment_epoch))
                for i, (images, target, _) in enumerate(tqdm(dataloader)):
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
            augment_epoch = self.cfg["augment_epoch"]
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
            alpha = self.cfg["alpha"]
        if beta is None:
            beta = self.cfg["beta"]
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
            return all_predictions, all_targets
        elif hasattr(self, "test_features") and hasattr(self, "test_labels"):
            if adapt:
                fused_logits = self._fuse_logits(
                    self.test_features,
                    self.test_clip_logits,
                    alpha=alpha,
                    beta=beta,
                )
                all_predictions = fused_logits.softmax(dim=-1).argmax(dim=1)
            else:
                all_predictions = self.test_clip_logits.softmax(dim=-1).argmax(dim=1)
            return all_predictions.cpu().numpy(), self.test_labels.cpu().numpy()

    def train_keys(self, dataloader, epoch=10, dataloader_eval=None):
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

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epoch * len(dataloader)
        )

        beta, alpha = self.cfg["beta"], self.cfg["alpha"]
        best_acc, best_epoch = 0.0, 0

        for train_idx in range(epoch):
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print("Train Epoch: {:} / {:}".format(train_idx, epoch))

            for i, (images, target, _) in enumerate(tqdm(dataloader)):
                images, target = images.to(self.device), target.to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                affinity = adapter(image_features)
                cache_logits = (
                    (-1) * (beta - beta * affinity)
                ).exp() @ self.cache_values
                clip_logits = 100.0 * image_features @ self.text_features.t()
                fused_logits = clip_logits + cache_logits * alpha

                loss = F.cross_entropy(fused_logits, target)
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

            adapter.eval()
            affinity = adapter(test_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
            clip_logits = 100.0 * test_features @ self.text_features.t()
            fused_logits = clip_logits + cache_logits * alpha
            probs = fused_logits.softmax(dim=-1)
            pred_label = probs.argmax(dim=1)
            accuracy = metrics.accuracy_score(
                test_labels.cpu().numpy(), pred_label.cpu().numpy()
            )
            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = train_idx
        print(
            f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n"
        )

    def search_hp(
        self,
        dataloader=None,
        search_scale=[50, 50],
        search_step=[200, 20],
        inplace=True,
    ):
        if dataloader is not None:
            features, labels = self.pre_load_features(dataloader=dataloader)
        elif hasattr(self, "test_features") and hasattr(self, "test_labels"):
            features, labels = self.test_features, self.test_labels
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
            for alpha in alpha_list:
                cache_logits = (
                    (-1) * (beta - beta * affinity)
                ).exp() @ self.cache_values

                fused_logits = clip_logits + cache_logits * alpha
                probs = fused_logits.softmax(dim=-1)
                pred_label = probs.argmax(dim=1)
                accuracy = metrics.accuracy_score(
                    labels.cpu().numpy(), pred_label.cpu().numpy()
                )

                if accuracy > best_acc:
                    print(
                        "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(
                            beta, alpha, accuracy
                        )
                    )
                    best_acc = accuracy
                    best_beta = beta
                    best_alpha = alpha
        if inplace:
            self.cfg["alpha"] = best_alpha
            self.cfg["beta"] = best_beta
        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

        return best_beta, best_alpha

    def _fuse_logits(self, image_featrues, clip_logits, alpha, beta):
        with torch.no_grad():
            if hasattr(self, "cache_keys") and hasattr(self, "cache_values"):
                affinity = image_featrues @ self.cache_keys
                cache_logits = ((-1) * beta * (1 - affinity)).exp() @ self.cache_values
                tip_logits = clip_logits + cache_logits * alpha
                return tip_logits
            else:
                warnings.warn("No Cached Error, Turn to Plain Mode")
                return clip_logits

    @torch.no_grad()
    def __call__(self, images, adapt=True, alpha=None, beta=None):
        if alpha is None:
            alpha = self.cfg["alpha"]
        if beta is None:
            beta = self.cfg["beta"]
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
