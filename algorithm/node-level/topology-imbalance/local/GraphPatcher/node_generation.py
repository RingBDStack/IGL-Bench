import torch 
from torch.utils.data import DataLoader

from src.model import Generator, GCN, SAGE, GAT
from src.option import OptionsGenerator
from src.data_train import Graph_Dataset, Graph_Dataset_splits, Graph_Collator_infer, Graph_Collator_train
from src.data import load_data, preprocess
from src.utils import kl_div, init_logger, accuracy, f1_macro, inject_nodes, evaluate_generator
from tqdm import tqdm
import pickle 
import timeit
import time

def train(GNN, generator, dataloader, args, device):

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accumulate_counter = 0
    accumulated_loss = 0
    accumulated_loss_ = 0
    optimizer.zero_grad()
    best_val_entropy = 100
    best_val_acc = 0
    test_acc = 0
    best_generation_step = 0
    patience = 0
    generator.train()
    current_iteration = 0

    while current_iteration < args.training_iteration:
        # Training Loops
        start = timeit.default_timer()
        for data in tqdm(dataloader) if args.bar else dataloader:
            # Starting GraphPatcher Training 
            batched_graphs, inverse_indices = data[0], data[1]
            batched_graphs = [bg.to(device) for bg in batched_graphs]
            starting_graphs = batched_graphs[0]
            generation_targets = batched_graphs[1:]
            loss = 0
            # Iterative Patching  
            for generation_graph, this_inverse_indces in zip(generation_targets, inverse_indices[1:-1]):
                with torch.no_grad():
                    target_distribution = GNN(generation_graph, generation_graph.ndata['feat'])[this_inverse_indces]
                    target_distribution = target_distribution.reshape(args.batch_size, 10, -1)
                generated_neighbors = generator(starting_graphs, inverse_indices[0])
                starting_graphs = inject_nodes(starting_graphs, generated_neighbors, inverse_indices[0], device)
                reconstructed_distribution = GNN(starting_graphs, starting_graphs.ndata['feat'])[inverse_indices[0]]
                reconstructed_distribution = reconstructed_distribution.unsqueeze(1).expand_as(target_distribution)
                loss += kl_div(reconstructed_distribution, target_distribution)
                starting_graphs.ndata['feat'] = starting_graphs.ndata['feat'].detach()
            
            # Final Patching 
            with torch.no_grad():
                target_distribution = GNN(generation_targets[-1], generation_targets[-1].ndata['feat'])[inverse_indices[-1]]
            generated_neighbors = generator(starting_graphs, inverse_indices[0])
            starting_graphs = inject_nodes(starting_graphs, generated_neighbors, inverse_indices[0], device)
            reconstructed_distribution = GNN(starting_graphs, starting_graphs.ndata['feat'])[inverse_indices[0]]
            loss += kl_div(reconstructed_distribution, target_distribution)
            loss.backward()

            accumulate_counter += 1
            accumulated_loss += loss.item()
            accumulated_loss_ += loss.item()
            if accumulate_counter % args.accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.wandb:
                    wandb.log({'Running loss': accumulated_loss}, step=current_iteration)
                    accumulated_loss  = 0
                current_iteration += 1
                if current_iteration % args.eval_iteration == 0:
                    if current_iteration < args.warmup_steps:
                        logger.info(f"[Warming up {current_iteration}/{args.training_iteration}]: accumulated loss {round(accumulated_loss_, 4)}")
                        accumulated_loss_ = 0
                        continue
                    stop = timeit.default_timer()
                    logger.info(f"[Training {current_iteration}/{args.training_iteration}]: accumulated loss {round(accumulated_loss_, 4)}| Test ACC: {test_acc}| Time Consumed: {round(stop - start, 2)} s| All Time {round(stop - all_start, 2)} s")
                    accumulated_loss_ = 0   
                    # Validation Loop
                    generator.eval()
                    vals = evaluate_generator(dataloader_val, generator, GNN,\
                                        graph.ndata['label'][graph.ndata['val_mask'].nonzero().squeeze()], device, args, iteration=args.total_generation_iteration)
                    val_acc, val_entropy, val_f1, val_bacc, val_roc_auc = \
                        [val[0] for val in vals], [val[1] for val in vals], [val[2] for val in vals], [val[3] for val in vals], [val[4] for val in vals]
                    
                    current_best_generation_step = val_acc.index(max(val_acc)) if args.generation_iteration==-1 else args.generation_iteration-1
                    # current_best_generation_step = val_entropy.index(min(val_entropy)) if args.generation_iteration==-1 else args.generation_iteration-1

                    current_val_acc, current_val_entropy, current_val_f1, current_val_bacc, current_val_roc_auc  = vals[current_best_generation_step]
                    statement = current_val_acc >= best_val_acc
                    # statement = current_val_entropy <= best_val_entropy

                    if args.wandb: wandb.log({'val_acc':current_val_acc, 'val_entropy': current_val_entropy}, step=current_iteration)
                
                    if statement:
                        best_val_entropy = current_val_entropy
                        best_val_acc = current_val_acc
                        best_generation_step = current_best_generation_step
                        logger.info(f"[Best Val Found {current_iteration}/{args.training_iteration}]: Validation Entropy {round(best_val_entropy, 4)}, Starting Testing..")
                        tests = evaluate_generator(dataloader_test, generator, GNN, \
                                        graph.ndata['label'][graph.ndata['test_mask'].nonzero().squeeze()], device, args, return_dist=True, iteration=args.total_generation_iteration)
                        test_acc, _, test_dist, test_f1, test_bacc, test_roc_auc = tests[best_generation_step]
                        if args.wandb: wandb.log({'test_acc':test_acc, 'diff_step':best_generation_step}, step=current_iteration)
                        pickle.dump(tests[best_generation_step][2].cpu(), open(f'outputs/{args.dataset}_{args.type}_{args.seed}.output'.format(dir), 'wb'))
                        logger.info(f"[Testing {current_iteration}/{args.training_iteration}]: Testing ACC {[round(test[0],6) for test in tests]} "
                                    f"| {[round(test[0],4) for test in tests][best_generation_step]} generation Step {best_generation_step+1}"
                                    f"| Testing ROC_AUC {[round(test[5],4) for test in tests]} | Testing bACC {[round(test[4],4) for test in tests]}"
                                    f"| Testing f1 {[round(test[3],4) for test in tests]}")
                        patience = 0
                    else:
                        patience += 1
                    if patience == args.patience:
                        logger.info(f"Early Stopping with Test Acc: {test_acc} | Testing ROC_AUC {test_roc_auc} | Testing bACC {test_bacc}"
                                    f"| Testing f1 {test_f1}| generation Step {best_generation_step+1}")
                        current_iteration = args.training_iteration
                        break
                    generator.train()
                    start = timeit.default_timer()
if __name__ == "__main__":
    option = OptionsGenerator()
    args = option.parse()
    print(args)
    if args.wandb:
        import wandb
        # Replace this with your own credentials if using wandb
        wandb.init(project="", entity='',
                name=f'{args.dataset}-{args.hid_dim}-{args.degree_train}-{args.drop_ratio}-{args.lr}-{args.batch_size*args.accumulate_step}')

    logger = init_logger('{}/{}_run_node_generator.log'.format(args.save_dir, args.dataset))
    device = torch.device("cuda:{}".format(args.device))
    ckpt_dir = f'{args.save_dir}/{args.dataset}_{args.type}' + (f'_{args.target_gnn}' if args.target_gnn != '' else '')
    model_params = pickle.load(open(f'{ckpt_dir}.config', 'rb'))
    graph = load_data(args.dataset, t=args.type, split='public', preprocess_=False if args.dataset=='arxiv' else True)
    preprocessed_graph = preprocess(graph)
    model = GCN(**model_params)
    
    model.load_state_dict(torch.load(ckpt_dir+'.ckpt'))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        original_dist= model(preprocessed_graph, preprocessed_graph.ndata['feat'])[graph.ndata['test_mask'].nonzero().squeeze()]
    acc = accuracy(original_dist, graph.ndata['label'][graph.ndata['test_mask'].nonzero().squeeze()])
    f1 = f1_macro(original_dist, graph.ndata['label'][graph.ndata['test_mask'].nonzero().squeeze()])

    model = model.to(device)
    generator = Generator(args.dropout, model.hidden_lst[0], args.hid_dim, model.hidden_lst[0], args, 
                          three_layer=args.three_layer, norm=args.norm, mp_norm=args.mp_norm).to(device)
    logger.info(generator)
    logger.info(args)
    dataset = Graph_Dataset(graph, args.degree_train, args.drop_ratio, 10, args.k)
    dataset_val = Graph_Dataset_splits(preprocessed_graph, graph.ndata['val_mask'].nonzero().squeeze(), args.k)
    dataset_test = Graph_Dataset_splits(preprocessed_graph, graph.ndata['test_mask'].nonzero().squeeze(), args.k)
    collator_train = Graph_Collator_train()
    collator_inference = Graph_Collator_infer()

    dataloader = DataLoader(dataset=dataset, drop_last=True,\
                            batch_size=args.batch_size, collate_fn=collator_train, num_workers=args.workers)
    dataloader_val = DataLoader(dataset=dataset_val, shuffle=False, drop_last=False,  \
                            batch_size=args.batch_size*4, collate_fn=collator_inference, num_workers=int(args.workers))
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=False, drop_last=False,\
                            batch_size=args.batch_size*4, collate_fn=collator_inference, num_workers=int(args.workers))

    logger.info("Total Number of Training Nodes: {}; Average Degree: {}".format(dataset.end, dataset.avg_degree))
    logger.info(f"Testing before Node Generation, ACC: {acc}, F1:{f1}")
    if not args.bar: logger.info('Running in the verbose mode!')

    torch.cuda.reset_max_memory_allocated()

    all_start = timeit.default_timer()
    start_time = time.time()
    train(model, generator, dataloader, args, device)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    max_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {max_memory / (1024**2):.4f} MB")