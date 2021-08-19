from utils import *
from loss import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from config import InitLearningRate,Args

from train import startTraining


def EntryPoint(mode='training',cuda=1,gpu=0):
    test_number = 10
    work_dir = os.getcwd()
    nn_model = 'GCN'
    data_dir = os.path.join(work_dir, 'data/ComboSampleData/')
    ##variables to store final result
    modularity_scores_gcn = {}
    nmi_gcn={}
    nmi={}
    C_init={}
    C_out = {}
    C_out_combo = {}
    graph_type = {}
    n_communities={}
    initial_partition_approach={}
    modularity_scores_combo = {}
    modularity_scores_combo_restricted={}
    loss={}
    model_parameter = {}
    data_name = []
    modularity_scores_classic={}





    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file[-3:] == 'mat':
                #append dataset name list
                # if 'karate_34' not in file:
                #     continue
                dataset = file[:-4]
                data_name.append(dataset )
                G = loadNetworkMat(file, data_dir)
                if nx.classes.function.is_directed(G):
                    graph_type[file] = 'directed'
                    initial_partition_approach[file]='LPA'
                else:
                    graph_type[file] = 'undirected'
                    initial_partition_approach[file]='Louvain'
                print(file, graph_type[file])
                if mode == 'training':
                    '''
                    {
                            'lr': init_lr.get_init_lr(dataset),
                            'lr_mode': lr_mode,
                            'grad_direction': grad_direction,
                            'nn_model': nn_model,
                            'n_epochs':30000,
                            'step_size':15000//100,
                            'cuda': cuda,
                            'cache_middle_result':cache_middle_result,
                            'early_stop':True
                        }
                    '''
                    lr = InitLearningRate(dataset)
                    # construct args as training parameter
                    args = Args(dataset=dataset)
                    args.setArgs(cuda=cuda,
                                 grad_direction=-1,
                                 nn_model=nn_model)
                    args.setArgs(
                        learning_rate=lr.get_init_lr(),
                        lr_mode=mode,
                        n_epochs=15000,
                        step_size=100,
                    )
                elif mode=='scanning':
                    lr = InitLearningRate(dataset)
                    # construct args as training parameter
                    args = Args(dataset=dataset)
                    args.setArgs(cuda=cuda,
                                 grad_direction=-1,
                                 nn_model=nn_model)
                    args.setArgs(
                        learning_rate=lr.get_init_lr(),
                        lr_mode=mode,
                        n_epochs=150,
                        step_size=1,
                    )


                modularity_scores_combo[file], partition = getNewComboPartition(G)
                #C_out_combo[file] = partition_to_binary_attachment(partition)
                modularity_scores_gcn[file], loss[file], C_out[file], model_parameter[file], C_init[file], \
                modularity_scores_classic[file], n_communities[file] = startTraining(nx_g=G,data_dir=data_dir,dataset=dataset,args=args.getArgs())
                modularity_scores_combo_restricted[file], partition = getNewComboPartition(G, maxcom=n_communities[file])
                nmi_gcn[file]=NMI(C_out[file],C_init[file])
                nmi[file] = NMI(C_out_combo[file], C_init[file])

    ##save log
    save_result(data_name, graph_type, modularity_scores_gcn, modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)
    #<network>,<initial partition approach>,<number of communities>,<initial modularity>,<modularity after fune-tuning>,<COMBO modularity >,<COMBO modularity without restricting the number of communities and the optimal number of communities it returns >
    save_result_for_report(data_name, initial_partition_approach,n_communities, modularity_scores_gcn, loss,modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)

    print('something')

def temp():
    '''
        test_number = 10
    work_dir = os.getcwd()
    nn_model = 'GCN'
    data_dir = os.path.join(work_dir, 'data/ComboSampleData/')
    # G = loadNetworkMat('karate_34.mat',data_dir)
    G = loadNetworkMat('celeganmetabolic_453.mat', data_dir)
    modularity_scores_gcn = {}
    nmi_gcn={}
    nmi={}
    C_init={}
    C_out = {}
    C_out_combo = {}
    graph_type = {}
    n_communities={}
    initial_partition_approach={}
    modularity_scores_combo = {}
    modularity_scores_combo_restricted={}
    loss={}
    model_parameter = {}
    data_name = []
    modularity_scores_classic={}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # print(file[-3:])
            # if 'celegansneural_297' not in file:
            #     continue
            if file[-3:] == 'mat':
                #append dataset name list
                # if 'karate_34' not in file:
                #     continue
                data_name.append(file)
                G = loadNetworkMat(file, data_dir)
                if nx.classes.function.is_directed(G):
                    graph_type[file] = 'directed'
                    initial_partition_approach[file]='LPA'
                else:
                    graph_type[file] = 'undirected'
                    initial_partition_approach[file]='Louvain'
                print(file, graph_type[file])

                # need to figure out it is weighted or not
                modularity_scores_combo[file], partition = getNewComboPartition(G)

                C_out_combo[file] = partition_to_binary_attachment(partition)

                #modularity_score, C_hat, model_structure, features, n_classes

                modularity_scores_gcn[file], loss[file],C_out[file], model_parameter[file],C_init[file],modularity_scores_classic[file], n_communities[file] = main(G, nn_model,grad_direction=-1,data_dir=data_dir,dataset=file[:-4],args)
                modularity_scores_combo_restricted[file], partition = getNewComboPartition(G, maxcom=n_communities[file])
                nmi_gcn[file]=NMI(C_out[file],C_init[file])
                nmi[file] = NMI(C_out_combo[file], C_init[file])



    ##save log
    save_result(data_name, graph_type, modularity_scores_gcn, modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)
    #<network>,<initial partition approach>,<number of communities>,<initial modularity>,<modularity after fune-tuning>,<COMBO modularity >,<COMBO modularity without restricting the number of communities and the optimal number of communities it returns >
    save_result_for_report(data_name, initial_partition_approach,n_communities, modularity_scores_gcn, loss,modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)

    print('something')
    :return:
    '''

if __name__ == "__main__":
    cuda=1
    gpu=0
    EntryPoint(mode='training',cuda=cuda,gpu=gpu)






