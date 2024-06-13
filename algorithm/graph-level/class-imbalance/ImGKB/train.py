import argparse
import time
import numpy as np
import torch.nn as nn
import torch
import random
from torch import optim
from src.model import KGIB
from src.utils import load_data, generate_batches,  AverageMeter, compute_metrics,my_load_data,my_metric,compute_metrics_multiclass,test_multiclass
from sklearn.model_selection import StratifiedShuffleSplit
from src.loss import Loss
import csv
import warnings
import os
import yaml
from data import shuffle
warnings.filterwarnings("ignore")
def seed_torch(seed=23):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	torch.backends.cudnn.deterministic=True


def train(adj, features, graph_indicator, y, model,optimizer, criterion, beta):
	optimizer.zero_grad()
	output, loss_mi = model(adj, features, graph_indicator)
	loss_train =  criterion(output, y)
	loss_train = loss_train + beta * loss_mi
	loss_train.backward()
	optimizer.step()
	return output, loss_train

def val(adj, features, graph_indicator ,model):
	output, loss_mi = model(adj, features, graph_indicator)
	return output
def test(adj, features, graph_indicator ,model):
	output, loss_mi = model(adj, features, graph_indicator)
	return output

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def main(args):

	print("----------------------------------------------starting training------------------------------------------------")
	model = KGIB(features_dim, args.hidden_dim, args.hidden_graphs, args.size_hidden_graphs,  args.nclass, args.max_step, args.num_layers, device,args.backbone).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0)

	criterion = Loss()
	best_acc = 0
	min_loss =0
	id_auc = []
	for epoch in range(args.epochs):
		start = time.time()
		model.train()
		train_loss = AverageMeter()
		train_auc = AverageMeter()


		# Train for one epoch
		for i in range(n_train_batches):
			output, loss = train(adj_train[i], features_train[i], graph_indicator_train[i], y_train[i], model, optimizer, criterion, args.beta)
			train_loss.update(loss.item(), output.size(0))
			output = nn.functional.softmax(output, dim=1)
			if args.dataset == 'COLLAB':
				auc, _, _ ,_,_=compute_metrics_multiclass(output.data, y_train[i].data)
			else:
				auc, _, _ ,_,_=compute_metrics(output.data, y_train[i].data)
			train_auc.update(auc,output.size(0))

		# Evaluate on val set
		model.eval()
		val_auc = AverageMeter()
		val_acc = AverageMeter()
		val_bacc = AverageMeter()
		val_recall = AverageMeter()
		val_f1score = AverageMeter()
		for i in range(n_val_batches):
			output = val(adj_val[i], features_val[i], graph_indicator_val[i], model)
			output = nn.functional.softmax(output, dim=1)
			with torch.no_grad():
				if args.dataset == 'COLLAB':
					auc, recal,f1,acc,bacc=compute_metrics_multiclass(output.data, y_val[i].data)
				else:
					auc, recal,f1,acc,bacc=compute_metrics(output.data, y_val[i].data)
			val_auc.update(auc, output.size(0))
			val_recall.update(recal, output.size(0))
			val_f1score.update(f1, output.size(0))
			val_acc.update(acc, output.size(0))
			val_bacc.update(bacc, output.size(0))

		# # Evaluate on validation set
		# # model.eval()
		# val_auc1 = AverageMeter()
		# val_recall1 = AverageMeter()
		# val_f1score1 = AverageMeter()

		# for i in range(n_test_batches):
		# 	output = val(adj_test[i], features_test[i], graph_indicator_test[i], model)
		# 	output = nn.functional.softmax(output, dim=1)
		# 	acc1, recal, f1 = compute_metrics(output.data, y_test[i].data)
		# 	val_auc1.update(acc1, output.size(0))
		# 	val_recall1.update(recal, output.size(0))
		# 	val_f1score1.update(f1, output.size(0))

		# Print results
		# print("epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "train_auc=", "{:.5f}".format(train_auc.avg),
		# 	  "val_auc=", "{:.5f}".format(val_auc.avg), "val_acc=", "{:.5f}".format(val_acc.avg), "val_bacc=", "{:.5f}".format(val_bacc.avg))


		# with  open('./Results/train.csv', 'a+', newline='', encoding='utf-8') as csvFile:
		# 	writer = csv.writer(csvFile)
		# 	# 先写columns_name
		# 	writer.writerow((args.num_layers, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim, args.beta, Fold_idx,  epoch + 1,
		# 					 train_loss.avg, train_auc.avg, val_auc.avg.item(), val_recall.avg,
		# 					 val_f1score.avg, val_auc1.avg.item(), val_recall1.avg, val_f1score1.avg, time.time() - start))

		# Remember best auc
		best_epoch_auc=500
		is_best = val_auc.avg > best_acc
		best_acc = max(val_auc.avg, best_acc)
		if is_best :
			best_epoch_auc = epoch + 1
			id_auc.append(best_epoch_auc)
			if len(id_auc) > 10:
				id_auc = id_auc[1:]
			torch.save({
				'epoch': best_epoch_auc,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, 'save_model/{}/model_best{:d}.pth.tar'.format(args.dataset, best_epoch_auc))
			
			


	print("finished!","val_auc=", "{:.5f}".format(best_acc))
	# Testing 
	model.eval()
	best_epoch = id_auc[-1]
	test_auc = AverageMeter()
	test_recall = AverageMeter()
	test_f1score = AverageMeter()
	test_acc = AverageMeter()
	test_bacc = AverageMeter()
	checkpoint = torch.load('save_model/{}/model_best{:d}.pth.tar'.format(args.dataset, best_epoch))
	model.load_state_dict(checkpoint['state_dict'])
	print("testing epoch:","{:d}".format(checkpoint['epoch']))
	#optimizer.load_state_dict(checkpoint['optimizer'])
	logits=[]
	for i in range(n_test_batches):
		output = test(adj_test[i], features_test[i], graph_indicator_test[i], model)
		output = nn.functional.softmax(output, dim=1)
		logits.append(output.cpu())
	logits = torch.cat(logits)
	label = [t.cpu().numpy() for t in y_test]
	label = np.concatenate(label)
	if args.dataset =='COLLAB':
		auc, recal, f1,acc,bacc = test_multiclass(logits, label)
	else:
		auc, recal, f1,acc,bacc = my_metric(logits, label)
	test_auc.update(auc, output.size(0))
	test_recall.update(recal, output.size(0))
	test_f1score.update(f1, output.size(0))
	test_acc.update(acc, output.size(0))
	test_bacc.update(bacc, output.size(0))
	del model

	best_acc = test_acc.avg
	best_bacc = test_bacc.avg
	best_f1 = test_f1score.avg
	best_auc = test_auc.avg
	# print("AUC Loading checkpoint!", "test_auc=", "{:.5f}".format(best_auc), "test_recall=",
	# 	  "{:.5f}".format(best_recal), "test_f1score=", "{:.5f}".format(best_f1))
	print("The {:d}th Performance: ".format(Fold_idx))
	print("acc:","{:.4f}".format(best_acc),"bacc:","{:.4f}".format(best_bacc),"mf1:","{:.4f}".format(best_f1),"auroc:","{:.4f}".format(best_auc))
	

	L1acc.append(best_acc)
	L1bacc.append(best_bacc)
	L1f1.append(best_f1)
	L1auc.append(best_auc)

def save_to_yaml(args):
    file_name = f"./config/{args.dataset}/{args.backbone}.yml"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)

if __name__ == "__main__":
	# Argument parser
	parser = argparse.ArgumentParser(description='ImGKB')
	parser.add_argument('--dataset', default='MCF-7', help='Dataset name')
	parser.add_argument('--backbone', default='gcn', help='backbone')
	parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
	parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='Input batch size for training')
	parser.add_argument('--epochs', type=int, default=500, metavar='N', help='Number of epochs to train')
	parser.add_argument('--hidden-graphs', type=int, default=6, metavar='N', help='Number of hidden graphs')
	parser.add_argument('--size-hidden-graphs', type=int, default=6, metavar='N', help='Number of nodes of each hidden graph')
	parser.add_argument('--hidden-dim', type=int, default=96, metavar='N', help='Size of hidden layer of NN')
	parser.add_argument('--feature-dim', type=int, default=10, metavar='N', help='Input size')
	parser.add_argument('--num-layers', type=int, default=2, metavar='N', help='Number of layer of KerGAD')
	parser.add_argument('--penultimate-dim', type=int, default=32, metavar='N', help='Size of penultimate layer of NN')
	parser.add_argument('--max-step', type=int, default=4, metavar='N', help='Max length of walks')
	parser.add_argument('--nclass', type=int, default=2, metavar='N', help='Class number')
	parser.add_argument('--beta', type=float, default=0.3, metavar='beta', help='Compression coefficient')
	parser.add_argument('--normalize', action='store_true', default=True, help='Whether to normalize the kernel values')
	parser.add_argument('--graph_pooling_type', type=str, default='average', choices=["sum", "average"], help='the type of graph pooling (sum/average)')
	parser.add_argument('--n_split', type=int, default=10, help='cross validation')
	parser.add_argument('--seed', type=int, default=6, help='random seed')
	parser.add_argument('--imb', type=float, default=0.9, help='imb_ratio')
	args = parser.parse_args()
	save_to_yaml(args)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	adj_lst, features_lst, graph_labels,train_num,val_num = my_load_data(args.dataset,args.imb)
	N = len(adj_lst)
	features_dim = features_lst[0].shape[1]
	n_classes = np.unique(graph_labels).size
	args.feature_dim = features_dim
	args.nclass = n_classes

	#seed_torch(args.seed)
	#skf = StratifiedShuffleSplit(args.n_split, test_size=0.1, train_size=0.9, random_state=args.seed)
	L1auc = list()
	L1recall = list()
	L1f1 = list()
	L1acc = list()
	L1bacc = list()
	Fold_idx = 0
	for i in range (10):
		set_seed(args.seed+Fold_idx)

		train_index, val_index, test_index = shuffle(N, train_num, val_num, graph_labels)

		adj_train = [adj_lst[i] for i in train_index]
		feats_train = [features_lst[i] for i in train_index]
		label_train = [graph_labels[i] for i in train_index]


		adj_val = [adj_lst[i] for i in val_index]
		feats_val = [features_lst[i] for i in val_index]
		label_val = [graph_labels[i] for i in val_index]

		adj_test = [adj_lst[i] for i in test_index]
		feats_test = [features_lst[i] for i in test_index]
		label_test = [graph_labels[i] for i in test_index]

		adj_train, features_train, graph_pool_lst1, graph_indicator_train, y_train, n_train_batches = generate_batches(
			adj_train, feats_train, label_train, args.batch_size, args.graph_pooling_type, device)

		adj_val, features_val, _, graph_indicator_val, y_val, n_val_batches = generate_batches(
			adj_val, feats_val, label_val, args.batch_size, args.graph_pooling_type, device)

		adj_test, features_test, graph_pool_lst2, graph_indicator_test, y_test, n_test_batches = generate_batches(
			adj_test, feats_test, label_test, args.batch_size, args.graph_pooling_type, device)

		main(args)
		Fold_idx += 1
	print("acc_mean=", "{:.4f}".format(np.mean(L1acc))," acc_std=", "{:.4f}".format(np.std(L1acc)))
	print("bacc_mean=", "{:.4f}".format(np.mean(L1bacc))," bacc_std=", "{:.4f}".format(np.std(L1bacc)))
	print("mf1_mean=", "{:.4f}".format(np.mean(L1f1))," mf1_std=", "{:.4f}".format(np.std(L1f1)))
	print("auc_mean=", "{:.4f}".format(np.mean(L1auc))," auc_std=", "{:.4f}".format(np.std(L1auc)))

	file_path = f'res/{args.dataset}.txt'

	with open(file_path, 'a') as file:
		file.write("acc_mean= {:.4f} acc_std= {:.4f}\n".format(np.mean(L1acc), np.std(L1acc)))
		file.write("bacc_mean= {:.4f} bacc_std= {:.4f}\n".format(np.mean(L1bacc), np.std(L1bacc)))
		file.write("mf1_mean= {:.4f} mf1_std= {:.4f}\n".format(np.mean(L1f1), np.std(L1f1)))
		file.write("auc_mean= {:.4f} auc_std= {:.4f}\n\n".format(np.mean(L1auc), np.std(L1auc)))


	# with  open('./Results/final.csv', 'a+', newline='', encoding='utf-8') as csvFile:
	# 	writer = csv.writer(csvFile)
	# 	writer.writerow((args.num_layers, args.max_step, args.hidden_graphs,
	# 					 args.size_hidden_graphs, args.hidden_dim,
	# 					 args.beta, np.mean(L1auc), np.mean(L1recall),
	# 					 np.mean(L1f1score)))

