
import torch.optim.lr_scheduler
from loss import *






def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q,modularity_classic, args):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    n_hidden = features.shape[1]  # number of hidden nodes
    n_layers = 0  # number of hidden layers
    self_loop = True  #
    visualize_model = False
    last_score = 0
    precision= 1e-8

    grad_direction = args['grad_direction']

    lr = args['lr']
    cuda = args['cuda']
    nn_model = args['nn_model']
    cache_middle_result=args['cache_middle_result']
    n_epochs=args['n_epochs']
    step_size = args['step_size']

    if 'early_stop' in args.keys():
        early_stop=args['early_stop']
    else:
        early_stop=False
    #early_stop=False


    # step_size=int(n_epochs/100)

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

    if visualize_model:
        print_parameter(model)
        print_parameter(loss_fcn)

    if n_layers==0:
        W1_grad = []
        W2_grad = []
        bias_grad=[]
        W1 = []
        W2 = []
        bias = []
        W1_name = []
        W2_name = []
        bias_name= []
        Y_grad = []
        def grad_W2(grad):
            W2_grad.append(grad)

        def grad_W1(grad):
            W1_grad.append(grad)
        def grad_bias(grad):
            bias_grad.append(grad)

        for name, param in model.named_parameters():
            print(name,param)

            if 'weight_1' in name:
                param.register_hook(grad_W1)
                W1_name.append(name)
                W1.append(param)
            if 'weight_2' in name:
                param.register_hook(grad_W2)
                W2_name.append(name)
                W2.append(param)
            if 'bias' in name:
                param.register_hook(grad_bias)
                bias_name.append(name)
                bias.append(param)


    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    StepLR= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       factor=0.1,
                                                       patience=2,
                                                       verbose=True,
                                                       threshold=0.0001,
                                                       threshold_mode='rel',
                                                       cooldown=0,
                                                       min_lr=1e-10,
                                                       eps=1e-10
                                                       )
    #lr= factor *lr

    # apply weight_decay scheduler
    # use self written D method
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=weight_decay_gamma)
    # optimizer = mySGD(model.parameters(), lr=lr, batch_size=features.shape[0], grad_direction=grad_direction)
    # train and evaluate (with modularity score and labels)
    dur = []
    M=[]

    print("initial inputs \n", features)
    print("#####################")

    for epoch in range(n_epochs):

        model.train()
        t0 = time.time()
        C_hat = model(features)
        # print('#############WX###################')
        # print(C_hat)
        # use train_mask to train
        loss = loss_fcn(C_hat[mask])
        if epoch>0 and early_stop:
            if optimizer.param_groups[0]['lr']<precision:
                print('loss less than 1e-10 training end')
                print(
                    "Epoch {} | Time(s) {} |  True_Modularity {} | Ground_Truth_Modulairty {} | ETputs(KTEPS) {}".format(
                        epoch,
                        np.mean(
                            dur),
                        grad_direction*(loss),
                        modularity_classic,
                        n_edges / np.mean(
                            dur) / 1000))
                break
        if epoch == 0:
            print("initial output WX : \n", C_hat)
            print('initial parameters and gradients')
            for i, p in enumerate(list(model.parameters())):
                print("param {}".format(i), p)
                print("param {} grad".format(i), p.grad)
            print('#####################')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step(grad_direction*(loss))


        dur.append(time.time() - t0)
        if epoch % step_size == 0:
            # if epoch % 1 == 0:
            # if visualize_model:
            #     print_parameter(model)
            #     print_parameter(loss_fcn)
            C_out, eval_loss = evaluate_M(C_hat, Q, cuda)
            print(
                "Epoch {} | Time(s) {} |  True_Modularity {}"
                " | Ground_Truth_Modulairty {} | ETputs(KTEPS) {}".format(epoch,
                                                                    np.mean(dur),
                                                                     grad_direction*(loss),
                                                                     modularity_classic,
                                                                     n_edges / np.mean(
                                                                                                                     dur) / 1000))
        if cache_middle_result:
            M.append(-loss.item())



        last_score = loss
        last_C_hat = C_hat

    C_init, modularity_init = evaluate_M(features, Q, cuda)

    print('initial modularity is', modularity_init)
    print(C_hat)
    C_hat, modularity_score = evaluate_M(C_hat, Q, cuda)
    if torch.isnan(modularity_score):
        print('use objective function as output')
        modularity_score = grad_direction*loss

    return modularity_score.cpu().detach().numpy(), loss.item(),C_hat.cpu(), model.__str__(), features.cpu(),M

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
            modularity_score,loss ,C_hat, model_structure, features, middle_result[args['lr']] = \
                train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, modularity_classic,args)
            if i == len(lr_range) - 1:
                save_middle_result(middle_result,data_dir,dataset)
                return modularity_score,loss, C_hat, model_structure, features, modularity_classic, n_classes
    if args['lr_mode'] =='training':
        print('all initial rating tuned, now start to learn')
        #modularity_score.cpu().detach().numpy(), C_hat.cpu(), model.__str__(), features.cpu(),M
        modularity_score,loss ,C_hat, model_structure, features, middle_result[args['lr']]= \
            train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, modularity_classic,args)
        return modularity_score,loss, C_hat, model_structure, features, modularity_classic, n_classes
