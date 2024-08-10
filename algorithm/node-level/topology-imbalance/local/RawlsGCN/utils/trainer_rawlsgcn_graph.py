import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression

from utils.evaluator import Evaluator
from utils.utils import accuracy, tensor2matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreProcessingTrainer:
    def __init__(self, configs, data, model, on_gpu, device):
        self.configs = self.default_configs()
        self.configs.update(configs)
        self.data = data
        self.model = model
        self.device = device
        
        if on_gpu:
            self.data.graph = self.data.graph.to(device)
            self.data.features = self.data.features.to(device)
            self.data.labels = self.data.labels.to(device)
            self.data.train_idx = self.data.train_idx.to(device)
            self.data.val_idx = self.data.val_idx.to(device)
            self.data.test_idx = self.data.test_idx.to(device)
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
        best_val_acc = 0.0
        for epoch in range(self.configs["num_epoch"]):
            logger.info("Epoch {epoch}".format(epoch=epoch))

            self.model.train()
            self.opt.zero_grad()

            # training
            output = self.model(self.data.features, self.data.graph)
            loss_train = self.criterion(
                output[self.data.train_idx], self.data.labels[self.data.train_idx]
            )
            loss_train.backward()
            self.opt.step()

            # validation
            self.model.eval()
            self.evaluator.eval(
                output=output[self.data.train_idx],
                labels=self.data.labels[self.data.train_idx],
                idx=self.data.train_idx,
                raw_graph=self.data.raw_graph,
                stage="train",
            )
            _, acc = self.evaluator.eval(
                output=output[self.data.val_idx],
                labels=self.data.labels[self.data.val_idx],
                idx=self.data.val_idx,
                raw_graph=self.data.raw_graph,
                stage="validation",
            )
            if acc > best_val_acc:
                best_val_acc = acc
                self.model.eval()
                output = self.model(self.data.features, self.data.graph)

                _, test_acc = self.evaluator.eval(
                    output=output[self.data.test_idx],
                    labels=self.data.labels[self.data.test_idx],
                    idx=self.data.test_idx,
                    raw_graph=self.data.raw_graph,
                    stage="test",
                )
            
            # self._save_model()
        print(test_acc)

    def test(self):
        self.model.eval()
        output = self.model(self.data.features, self.data.graph)

        folder_name = "embedding"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = os.path.join(folder_name, f"{self.data.dataset_name}_{self.configs['seed']}_{self.configs['type']}.pt")
        torch.save(output[self.data.test_idx], file_name)

        self.evaluator.eval(
            output=output[self.data.test_idx],
            labels=self.data.labels[self.data.test_idx],
            idx=self.data.test_idx,
            raw_graph=self.data.raw_graph,
            stage="test",
        )

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
            "loss": "negative_log_likelihood",
            "type": "mid"
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
