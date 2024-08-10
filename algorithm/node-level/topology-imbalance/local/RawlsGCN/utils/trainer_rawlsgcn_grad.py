import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression

from utils.evaluator import Evaluator
from utils.utils import tensor2matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InProcessingTrainer:
    def __init__(self, configs, data, model, on_gpu, device):
        self.configs = self.default_configs()
        self.configs.update(configs)

        self.data = data
        self.model = model

        if "ablation" in configs:
            if configs["ablation"] == "row":
                self.doubly_stochastic_graph = self.data.get_row_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "column":
                self.doubly_stochastic_graph = self.data.get_column_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "symmetric":
                self.doubly_stochastic_graph = self.data.get_symmetric_normalized(
                    tensor2matrix(self.data.graph.coalesce())
                )
            if configs["ablation"] == "doubly_stochastic":
                self.doubly_stochastic_graph = self.data.get_doubly_stochastic(
                    tensor2matrix(self.data.graph.coalesce())
                )
        else:
            self.doubly_stochastic_graph = self.data.get_doubly_stochastic(
                tensor2matrix(self.data.graph.coalesce())
            )

        self.on_gpu = on_gpu
        self.device = device
        if on_gpu:
            self.data.graph = self.data.graph.to(device)
            self.data.features = self.data.features.to(device)
            self.data.labels = self.data.labels.to(device)
            self.data.train_idx = self.data.train_idx.to(device)
            self.data.val_idx = self.data.val_idx.to(device)
            self.data.test_idx = self.data.test_idx.to(device)
            self.doubly_stochastic_graph = self.doubly_stochastic_graph.to(device)
            self.model.to(device)

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
        )

        if self.configs["loss"] == "negative_log_likelihood":
            self.criterion = nn.NLLLoss()
        elif self.configs["loss"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "loss in configs should be either `negative_log_likelihood` or `cross_entropy`"
            )

        self.evaluator = Evaluator(loss=self.configs["loss"], logger=logger)

    def train(self):
        logger.info(self.configs)
        for epoch in range(self.configs["num_epoch"]):
            logger.info("Epoch {epoch}".format(epoch=epoch))

            self.model.train()
            self.opt.zero_grad()

            # training
            pre_act_embs, embs = self.model(self.data.features, self.data.graph)
            loss_train = self.criterion(
                embs[-1][self.data.train_idx], self.data.labels[self.data.train_idx]
            )
            loss_train.backward()
            self._fix_gradient(pre_act_embs, embs)
            self.opt.step()

            # validation
            self.model.eval()
            self.evaluator.eval(
                output=embs[-1][self.data.train_idx],
                labels=self.data.labels[self.data.train_idx],
                idx=self.data.train_idx,
                raw_graph=self.data.raw_graph,
                stage="train",
            )
            self.evaluator.eval(
                output=embs[-1][self.data.val_idx],
                labels=self.data.labels[self.data.val_idx],
                idx=self.data.val_idx,
                raw_graph=self.data.raw_graph,
                stage="validation",
            )

            # self._save_model()

    def test(self):
        self.model.eval()
        _, embs = self.model(self.data.features, self.data.graph)

        folder_name = "embedding"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = os.path.join(folder_name, f"{self.data.dataset_name}_{self.configs['seed']}_{self.configs['type']}.pt")
        torch.save(embs[-1][self.data.test_idx], file_name)
        
        self.evaluator.eval(
            output=embs[-1][self.data.test_idx],
            labels=self.data.labels[self.data.test_idx],
            idx=self.data.test_idx,
            raw_graph=self.data.raw_graph,
            stage="test",
        )

    def _fix_gradient(self, pre_act_embs, embs):
        flag = 0  # flag = 0 for weight, flag = 1 for bias
        weights, biases = list(), list()
        # group params
        for name, param in self.model.named_parameters():
            layer, param_type = name.split(".")
            if param_type == "weight":
                if flag == 1:
                    flag = 0
                weights.append(param.data)
            else:
                if flag == 0:
                    flag = 1
                biases.append(param.data)
            flag = 1 - flag

        # fix gradient
        for name, param in self.model.named_parameters():
            layer, param_type = name.split(".")
            idx = self.model.layers_info[layer]
            # idx for embs and pre_act_embs are aligned here because we add a padding in embs (i.e., input features)
            if param_type == "weight":
                normalized_grad = torch.mm(
                    embs[idx].transpose(1, 0),
                    torch.sparse.mm(
                        self.doubly_stochastic_graph, pre_act_embs[idx].grad
                    ),
                )
            else:
                normalized_grad = torch.squeeze(
                    torch.mm(
                        torch.ones(1, self.data.num_nodes).to(self.device),
                        pre_act_embs[idx].grad,
                    )
                )
            param.grad = normalized_grad

    def _save_model(self):
        folder = "/".join(self.configs["save_path"].split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), self.configs["save_path"])

    @staticmethod
    def default_configs():
        configs = {
            "name": "cora",
            "model": "gcn",
            "num_epoch": 200,
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "type": "mid",
            "loss": "negative_log_likelihood",
        }
        configs["save_path"] = "ckpts/{name}/{model}/{setting}.pt".format(
            name=configs["name"],
            model=configs["model"],
            setting="lr={lr}_nepochs={nepochs}_decay={decay}".format(
                lr=configs["lr"],
                nepochs=configs["num_epoch"],
                decay=configs["weight_decay"],
            ),
        )
        return configs
