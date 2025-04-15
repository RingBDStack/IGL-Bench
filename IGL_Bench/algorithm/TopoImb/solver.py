from IGL_Bench.algorithm.TopoImb.topo_util import generate_topo_labels
import IGL_Bench.algorithm.TopoImb.utils as utils
from IGL_Bench.algorithm.TopoImb.model import StructGraphGNN
from IGL_Bench.algorithm.TopoImb.trainer import ReweighterGraphTrainer,TopoGraphTrainer,GClsTrainer
from IGL_Bench.backbone.gin import GIN_graph
from IGL_Bench.backbone.gcn import GCN_graph


from torch_geometric.loader import DataLoader as GeoDataLoader
import torch
import math

class TopoImb_graph_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        self.topo_labels = torch.tensor(generate_topo_labels(dataset, config),dtype=torch.int64,device=device)
        
        topo_size_dict = {}
        for topo in set(self.topo_labels.cpu().numpy()):
            topo_size = (self.topo_labels==topo).sum()
            topo_size_dict[topo] = topo_size.item()
        self.topo_size_dict = topo_size_dict
        
        self.model = {}
        self.prepare_data_loaders(device)        
        self.initialize_models()
        self.initialize_trainers()

    def prepare_data_loaders(self, device='cuda'):
        batch_size = getattr(self.config, 'batch_size', 128)

        train_data = self.dataset[self.dataset.train_mask]
        val_data = self.dataset[self.dataset.val_mask]
        test_data = self.dataset[self.dataset.test_mask]

        self.train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = GeoDataLoader(val_data, batch_size=batch_size, shuffle=True)
        self.test_loader = GeoDataLoader(test_data, batch_size=batch_size, shuffle=True)
                
    def initialize_models(self):
        if self.config.backbone == "GIN":
            self.model['default'] = GIN_graph(
                n_feat=self.dataset.num_features,
                n_hidden=self.config.hidden_dim,
                n_class=self.dataset.num_classes,
                n_layer=self.config.n_layer,
                dropout=self.config.dropout
            )
        elif self.config.backbone == "GCN":
            self.model['default'] = GCN_graph(
                n_feat=self.dataset.num_features,
                n_hidden=self.config.hidden_dim,
                n_class=self.dataset.num_classes,
                n_layer=self.config.n_layer,
                dropout=self.config.dropout
            )
        
        self.reweighter = StructGraphGNN(
            self.config,
            nfeat=self.dataset.num_features,
            nhid=self.config.hidden_dim,
            nclass=self.dataset.num_classes,
            dropout=0,
            n_mem=self.config.n_mem,
            nlayer=self.config.n_layer,
            use_key=self.config.use_key,
            att=self.config.att
        )
        
        self.model['default'] = self.model['default'].to(self.device)
        self.reweighter = self.reweighter.to(self.device)

    def initialize_trainers(self):
        self.trainers = []
        down_trainer = ReweighterGraphTrainer(
            self.config,
            self.model['default'],
            reweighter=self.reweighter,
            weight=self.config.reweight_weight,
            dataset=self.dataset
        )
        self.trainers.append(down_trainer)  
        
        self.re_trainers = []
        
        Graph_Retrainer_dict = {'cls': GClsTrainer, 'wlcls': TopoGraphTrainer}
        RetrainerClass = Graph_Retrainer_dict[self.config.re_task]
        
        retrainer = RetrainerClass(
            self.config,
            self.reweighter,
            dataset=self.dataset,
            out_size=len(set(self.topo_labels.cpu().numpy())),  
            shared_encoder=True
        )
        self.re_trainers.append(retrainer)
    
    def reset_parameters(self):
        self.initialize_models()
        self.initialize_trainers()
        
    def train(self):
        self.reset_parameters()
        
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 10)
        least_epoch = getattr(self.config, 'least_epoch', 200)
        best_val_acc = 0
        best_epoch = 0
        
        steps = math.ceil((len(self.dataset)/self.config.batch_size))
        
        for epoch in range(0, num_epochs + 1):
            #test
            if epoch % 50 == 0 and epoch <num_epochs:
                for ctrainer in self.trainers:
                    #test all data inside dataset
                    log_test = {}
                    for batch, data in enumerate(self.test_loader):
                        log_info = ctrainer.test(data.to(self.device))
                        for key in log_info:
                            if key not in log_test.keys():
                                log_test[key] = utils.meters(orders=1)
                            log_test[key].update(log_info[key], 1)                

                    print('test log in step{}: {}'.format(epoch*steps, {key: log_test[key].avg() for key in log_info}))             
                    
            for _trainer in self.trainers:#
                data_iter=0
                for batch,data in enumerate(self.train_loader):
                    log_info = _trainer.train_step(data.to(self.device), epoch)
                    if data.y.dim() > 1:
                        data.y = data.y.view(-1)
                    
                    if len(self.re_trainers) !=0:
                        for re_trainer in self.re_trainers:
                            topo_label_train = self.topo_labels[self.dataset.train_mask]
                            topo_label_train = topo_label_train[data_iter:data_iter+data.y.shape[0]]
                            data_iter = data_iter+data.y.shape[0]

                            log_info = re_trainer.train_step(data.to(self.device), epoch, label=topo_label_train)
                                
                for batch,data in enumerate(self.train_loader):
                    log_info = _trainer.train_mem_step(data.to(self.device), epoch) 
            #for eval:               
            with torch.no_grad():
                ctrainer = self.trainers[0] 
                for batch, data in enumerate(self.val_loader):
                    log_info = ctrainer.test(data.to(self.device))
                    for key in log_info:
                        if key not in log_test.keys():
                            log_test[key] = utils.meters(orders=1)
                        log_test[key].update(log_info[key], 1)                
            acc_val = log_test['acc_test'].avg()
            print(f"acc_val:{acc_val}")
            
            if best_val_acc < acc_val:
                best_val_acc = acc_val
                best_epoch = epoch
                
            if epoch - best_epoch >patience and epoch >least_epoch:
                print(f"Early stopping at epoch {epoch}.") 
                break    
        print("Training Finished!")
    
    def test(self):
        ctrainer = self.trainers[0]
        log_test = {}
        for batch, data in enumerate(self.test_loader):
            log_info = ctrainer.test(data.to(self.device))
            for key in log_info:
                if key not in log_test.keys():
                    log_test[key] = utils.meters(orders=1)
                log_test[key].update(log_info[key], 1)                
        acc_test = log_test['acc_test'].avg()
        bacc_test = log_test['bacc_test'].avg()
        roc_test = log_test['roc_test'].avg()
        macroF_test = log_test['macroF_test'].avg()
        
        return acc_test, bacc_test, macroF_test, roc_test