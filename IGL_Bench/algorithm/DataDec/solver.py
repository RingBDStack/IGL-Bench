from IGL_Bench.algorithm.DataDec.dataloader import MyDataset, IndexSampler
from IGL_Bench.algorithm.DataDec.model import *
from IGL_Bench.algorithm.DataDec.contrast import DualBranchContrast_diet
from IGL_Bench.algorithm.DataDec.prune import Mask
from torch_geometric.data import DataLoader
import copy
from torch_geometric.data import Data
import GCL.augmentors as A
import GCL.losses as L
import random
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize

def collate_fn(batch):
    data_list = [item[0] for item in batch]  
    indices = [item[1] for item in batch]  
    batch_data = Data.collate(data_list) 
    batch_data.indices = indices

    return batch_data, indices

class DataDec_graph_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        self.train_index = self.dataset.train_index.tolist()
        fine_tune_size = int(len(self.dataset.train_index) * config.fine_tune_ratio)
        self.fine_index = random.sample(self.dataset.train_index.tolist(), fine_tune_size)
        self.real_train_index = list(set(self.dataset.train_index.tolist()) - set(self.fine_index))
        
        self.sampled_list = copy.deepcopy(self.real_train_index)
        self.sampled_list_ep, self.unsampled_list_ep = copy.deepcopy(self.sampled_list), []
        
        self.prepare_datatsets()
        self.model = {}
        self.optimizer = {}
        self.initializtion()
        
    def prepare_datatsets(self):
        train_index = self.real_train_index
        val_index = self.dataset.val_index.tolist()
        test_index = self.dataset.test_index.tolist()
        
        full_index = self.train_index + val_index + test_index
        full_dataset = MyDataset(self.dataset, full_index)
        
        train_sampler = IndexSampler(train_index)
        self.train_loader = DataLoader(full_dataset, batch_size=self.config.batch_size, 
                                sampler=train_sampler, drop_last=False, pin_memory=True,
                                collate_fn=collate_fn)
        
        test_sampler = IndexSampler(test_index)
        self.test_loader = DataLoader(full_dataset, batch_size=self.config.batch_size, 
                                sampler=test_sampler, drop_last=False, pin_memory=True,
                                collate_fn=collate_fn)
    
        fine_sampler = IndexSampler(self.fine_index)
        self.fine_loader = DataLoader(full_dataset, batch_size=self.config.batch_size, 
                                sampler=fine_sampler, drop_last=False, pin_memory=True,
                                collate_fn=collate_fn)
        
    def initializtion(self):
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                            A.NodeDropping(pn=0.1),
                            A.FeatureMasking(pf=0.1),
                            A.EdgeRemoving(pe=0.1)], 1)
        if self.config.backbone == 'GIN':
            gconv = GINV(input_dim=self.dataset.num_features, hidden_dim=self.config.hidden_dim, 
                         num_layers=self.config.n_layer).to(self.device)
        elif self.config.backbone =='GCN':
            gconv = GCNV(input_dim=self.dataset.num_features, hidden_dim=self.config.hidden_dim, 
                         num_layers=self.config.n_layer).to(self.device)
        else:
            gconv = SAGE(input_dim=self.dataset.num_features, hidden_dim=self.config.hidden_dim, 
                         num_layers=self.config.n_layer).to(self.device)
            
        self.model['encoder'] = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(self.device)
        self.model['contrast'] = DualBranchContrast_diet(loss=L.InfoNCE(tau=0.2), mode='G2G').to(self.device)    
        
        self.optimizer['encoder'] = torch.optim.Adam(self.model['encoder'].parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)   

    def reset_parameters(self):
        self.initializtion()
    
    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epoch', 500)
        for epoch in range(0, num_epochs):
            data_prune_rate = 0.5 
            model_add_rate = cosine_schedule(epoch, num_epochs)
            self.prune_percent = self.config.random_prune_percent + (1 - model_add_rate) * (self.config.biggest_prune_percent - self.config.random_prune_percent) 
            
            loss, self.sampled_list_ep, self.unsampled_list_ep = self.train_epoch(self.model['encoder'], self.model['contrast'],
                                                                        self.real_train_index,  
                                                                        self.optimizer['encoder'], self.config,
                                                                        sampled_list=np.asarray(self.sampled_list_ep).astype(int),
                                                                        unsampled_list=np.asarray(self.unsampled_list_ep).astype(int),
                                                                        explore_rate=self.config.explore_rate, dataloader=self.train_loader,
                                                                        data_prune_rate=data_prune_rate, is_error_rank=self.config.is_error_rank)
            if epoch == 5:
                self.sampled_list.clear()
                self.sampled_list.extend(self.sampled_list_ep)
        
        self.model['classifier'] = self.fine_tune()
        
            
    def train_epoch(self, encoder_model, contrast_model,
            dataset,
            optimizer, args,
            sampled_list=None, unsampled_list=None,
            explore_rate=0.1, dataloader=None,
            data_prune_rate=0.5, is_error_rank=True):
        
        error_score_list = np.asarray([])
        grad_norm_list = np.asarray([])
        indices_list = np.asarray([])
        explore_num = int(explore_rate * len(dataset))
        diet_len = int(len(dataset) * data_prune_rate)

        
        encoder_model.train()
        epoch_loss = 0
        pruneMask = Mask(encoder_model)
        prunePercent = args.prune_percent
        randomPrunePercent = args.random_prune_percent
        magnitudePrunePercent = prunePercent - randomPrunePercent
        pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

        
        if sampled_list is None:
            sampled_list = np.asarray(list(range(len(dataset))))
        if unsampled_list is None:
            unsampled_list = np.asarray([]).astype(int)
        

        
        dataloader = list(dataloader)
        
        for (data, indices) in dataloader:
            data = data.to('cuda')
            optimizer.zero_grad()
            
            with torch.no_grad():
                
                encoder_model.eval()
                encoder_model.set_prune_flag(True)
                features_2 = encoder_model(data.x, data.edge_index, data.batch)
            encoder_model.train()
            encoder_model.set_prune_flag(False)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

            _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]

            loss, error_scores1, error_scores2, sample1, sample2 = contrast_model(g1=g1, g2=g2, batch=data.batch)
            loss.backward()

            grad_norm1 = get_grad_norm_fn(sample1.grad.cpu().numpy())
            grad_norm2 = get_grad_norm_fn(sample2.grad.cpu().numpy())
            grad_norm = grad_norm1 + grad_norm2
            error_scores = error_scores1 + error_scores2
            error_score_list = np.concatenate((error_score_list, error_scores))
            grad_norm_list = np.concatenate((grad_norm_list, grad_norm))
            indices_list = np.concatenate((indices_list, indices.tolist()))
            optimizer.step()
            epoch_loss += loss.item()

        
        error_score_sorted_id = np.argsort(-error_score_list)
        grad_norm_sorted_id = np.argsort(-grad_norm_list)
        error_rank_id = indices_list[error_score_sorted_id]

        
        if is_error_rank:
            rank_id = error_rank_id.astype(int)
        else:
            grad_rank_id = indices_list[error_score_sorted_id]
            rank_id = grad_rank_id.astype(int)
        keep_num = diet_len - explore_num
        kept_sampled_list = rank_id[:keep_num]
        removed_sampled_list = rank_id[keep_num:]
        unsampled_list = np.concatenate((unsampled_list, removed_sampled_list))
        np.random.shuffle(unsampled_list)
        newly_added_sampled_list = unsampled_list[:explore_num]
        unsampled_list = unsampled_list[explore_num:]
        sampled_list = np.concatenate((kept_sampled_list, newly_added_sampled_list))

        return epoch_loss/len(dataloader), sampled_list[::-1].tolist(), unsampled_list
    
    def fine_tune(self):
        self.model['encoder'].eval()
        dataloader = list(self.fine_loader)
        x = []
        y = []
        
        for data,indices in dataloader:
            data = data.to('cuda')
            with torch.no_grad():
                _, g, _, _, _, _ =self.model['encoder'](data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
            
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        
        # ps = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ps = KFold(n_splits=3, shuffle=True, random_state=42)
        params = {'C': [0.001,0.01, 0.1, 1, 10]}
        classifier = GridSearchCV(SVC(kernel='linear', probability=True), 
                                  params, cv=ps, scoring='accuracy', verbose=0,n_jobs=-1)
        
        classifier.fit(x.cpu().numpy(), y.cpu().numpy())
        print('Fine-tune acc:', classifier.best_score_)
        
        return classifier
    
    def test(self):
        self.model['encoder'].eval()
        dataloader = list(self.test_loader)
        x = []
        y = []
        
        for data,indices in dataloader:
            data = data.to('cuda')
            with torch.no_grad():
                _, g, _, _, _, _ =self.model['encoder'](data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
            
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        y_pred = self.model['classifier'].predict(x.cpu().numpy())
        
        acc = accuracy_score(y.cpu().numpy(), y_pred)
        bcc = balanced_accuracy_score(y.cpu().numpy(), y_pred)
        mf1 = f1_score(y.cpu().numpy(), y_pred, average='macro')
        if len(torch.unique(y)) == 2:  
            auc_roc = roc_auc_score(y.cpu().numpy(), self.model['classifier'].predict_proba(x.cpu().numpy())[:, 1])
        else:  
            y_bin = label_binarize(y.cpu().numpy(), classes=list(range(len(torch.unique(y)))))
            auc_roc = roc_auc_score(y_bin, self.model['classifier'].predict_proba(x.cpu().numpy()), average='macro', multi_class='ovr')
        
        return acc, bcc, mf1, auc_roc