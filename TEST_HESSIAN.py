
import torch.optim.lr_scheduler
from loss import *
from config import InitLearningRate,Args

def process_hess(hess):
    '''
    if there is a single input, this will be a single Tensor containing the Hessian for the input.
    If it is a tuple, then the Hessian will be a tuple of tuples where Hessian[i][j] will contain \
    the Hessian of the ith input and jth input with size the sum of the size of the ith input plus the size of the jth input.
    Hessian[i][j] will have the same dtype and device as the corresponding ith input.
    :param hess:
    :return:
    '''
    eig= []
    #didn't calculate the one with bias as it is not square matrix
    print('something')
    for i in range(len(hess)-1):
        for j in range(len(hess[0])-1):
            for i_w in range(len(hess[i][j])):
                for j_w in range(len(hess[i][j][0])):
                    #print(hess[i][j][i_w][j_w])
                    temp= hess[i][j][i_w][j_w].cpu()
                    e_values=np.linalg.eigvals(temp)
                    eig.append(e_values)
                    print(e_values)

    print(eig)


def loss_function_test(C,Q):
    Q=Q.float()
    C = C.cuda()
    Q = Q.cuda()

    Y=th.matmul(th.matmul(C.t(), Q), C)
    Z = Y.trace()
    return Z

def hessain(L,y,w):
    # if not isinstance(x,list):
    #     x = list(x)
    # #grad_Y = th.autograd.grad(y, x,  create_graph=True)
    dLdw = th.autograd.grad(L, w,   retain_graph=True,create_graph=True)
    #output of grad is a tuple size of number of params
    dLdw=dLdw[0]
    print(dLdw)
    hess_1 = th.zeros_like(dLdw)
    for i in range(dLdw.size(0)):
        for j in range(dLdw.size(1)):
            hess_1[i][j] =  th.autograd.grad(dLdw[i][j], w, retain_graph=True)[0][i][j]
    print(hess_1)
    # dLdy = th.autograd.grad(L, y,   retain_graph=True,create_graph=True)[0]
    #
    # dydw = th.zeros_like(y)
    # for i in range(dydw.size(0)):
    #     for j in range(dydw.size(1)):
    #         dydw[i][j] = th.autograd.grad(y[i][j], w, retain_graph=True, create_graph=True)[0][i][j]
    #
    # dydw_2 = th.zeros_like(dydw)
    # for i in range(dydw.size(0)):
    #     for j in range(dydw.size(1)):
    #         dydw_2[i][j] = th.autograd.grad(dydw[i][j], w, retain_graph=True, create_graph=True)[0][i][j]
    # dLdy_2 = th.zeros_like(dLdy)
    # for i in range(dLdy):
    #     for j in range(dLdy):
    #         dLdy_2[i][j] = th.autograd.grad(dLdy[i][j], y,   retain_graph=True,create_graph=True)[0][i][j]
    #
    # hess_2 = dLdy.t() @ dydw_2.t() + dLdy_2.t() @ [(dydw).pow(2)].t()
    # print(hess_2)





def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q,modularity_classic, args):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    n_hidden = features.shape[1]  # number of hidden nodes
    n_layers = 0  # number of hidden layers
    self_loop = True  #
    grad_direction = args['grad_direction']
    lr = args['lr']
    cuda = args['cuda']
    nn_model = args['nn_model']
    n_epochs=1
    if self_loop:
        g = dgl.add_self_loop(g)
    # run single train of some model
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        g = g.to('cuda:0')
        mask = mask.cuda()

    if nn_model == 'GCN':
        model = eval(nn_model)(g,
                               in_feats,
                               n_hidden,
                               n_classes,
                               n_layers,
                               F.relu,
                               dropout)
        if cuda:
            model.cuda()

    loss_fcn = ModularityScore(n_classes, cuda, grad_direction,Q)

    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("initial inputs \n", features)
    print("#####################")
    C_hat = th.zeros_like(features)
    C_hat.requires_grad=True
    C_hat.retains_grad=True
    for epoch in range(n_epochs):

        model.train()
        C_hat = model(features)
        loss = loss_fcn(C_hat[mask])

        # for param in model.parameters():
        #     print(param.grad)
        #     hessain(grad_direction*loss,C_hat,param)
        optimizer.zero_grad()

        #start to calculate the hessian, now only for 1 layers
        #T=th.autograd.functional.hessian(loss_fcn,C_hat,create_graph=True)
        def func(weight_1,weight_2,bias):
            print(model.layers[0].weight_1)
            del model.layers[0].weight_1
            if not getattr(model.layers[0],'weight_1',0):
                print('weight_1 deleted')
            model.layers[0].weight_1=weight_1
            print(model.layers[0].weight_1)
            del model.layers[0].weight_2
            model.layers[0].weight_2=weight_2
            del model.layers[0].bias
            model.layers[0].bias=bias
            Q1= Q
            temp= Q1.float()
            C = model(features)
            Q1 = temp.cuda()

            Y = th.matmul(th.matmul(C.t(), Q1), C)
            Z = Y.trace()
            return Z

        hess = th.autograd.functional.hessian(func, tuple(model.parameters()))
        print(hess)
        process_hess(hess)

        loss.backward()

        optimizer.step()
    return

def startTraining(nx_g,data_dir,dataset,args):
    lr_range = np.logspace(-7,2,num=10)
    g, features, n_classes, in_feats, n_edges, labels, Q, mask, modularity_classic =\
        generate_model_input(nx_g, args['cuda'])
    middle_result={}
    if args['lr_mode'] == 'scanning':
        print('learning_rate scanning mode for {} intervals from {} to {}'.
              format(len(lr_range), np.min(lr_range), np.max(lr_range)))
        for i, learning_rate in enumerate(lr_range):
            args['learning_rate']=learning_rate

            return train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, modularity_classic,args)
    if args['lr_mode'] =='training':
        print('all initial rating tuned, now start to learn')
        #modularity_score.cpu().detach().numpy(), C_hat.cpu(), model.__str__(), features.cpu(),M

        return train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, modularity_classic,args)


def EntryPoint(mode='training',cuda=1,gpu=0):
    test_number = 10
    work_dir = os.getcwd()
    nn_model = 'GCN'
    data_dir = os.path.join(work_dir, 'data/ComboSampleData/')
    ##variables to store final result

    data_name = []
    training_loops = 100


    for i in range(training_loops):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file[-3:] == 'mat':
                    #append dataset name list
                    if 'karate_34' not in file:
                        continue
                    dataset = file[:-4]
                    data_name.append(dataset )
                    G = loadNetworkMat(file, data_dir)
                    if mode == 'training':
                        lr = InitLearningRate(dataset,use_Adam=False,use_default_lr=False)
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
                            early_stop=True
                        )
                        startTraining(nx_g=G,data_dir=data_dir,dataset=dataset,args=args.getArgs())


    print('something')

if __name__ == "__main__":
    print(th.__version__)
    cuda=1
    gpu=0
    EntryPoint(mode='training',cuda=cuda,gpu=gpu)