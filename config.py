class InitLearningRate():
    def __init__(self,dataset,use_Adam=False,use_default_lr=True):
        self.init_lr = {
            'karate_34': 1,
            'jazz_198': 1,
            'lesmis_77': 0.1,
            'copperfield_112': 1e-4,
            'celeganmetabolic_453': 1e-2,
            'celegansneural_297': 1e-2,
            'email_1133': 1e-2,
            'USAir97_332': 1e-1,
            'USairports_1858': 0.1,
            'polbooks_105':1e-5
        }
        self.use_Adam=use_Adam
        self.use_default_lr = use_default_lr
        self.dataset = dataset


    def get_init_lr(self):
        if self.use_Adam or self.use_default_lr:
            return 1e-3
        return self.init_lr[self.dataset]

class Args():
    def __init__(self,dataset):
        self.init_lr = InitLearningRate(dataset)
        self.dataset=dataset
        self.args = {
            'lr': self.init_lr.get_init_lr(),
            'lr_mode': 'scanning',
            'grad_direction': 1,
            'n_epochs': 150,
            'nn_model': 'GCN',
            'step_size': 1,
            'cuda': 1,
            'cache_middle_result': True
        }
    def setArgs(self,
                learning_rate=1e-3,
                lr_mode = 'scanning',
                grad_direction=1,
                n_epochs=150,
                nn_model='GCN',
                step_size=1,
                cuda=1,
                cache_middle_result=True):
        self.args ={
            'lr': learning_rate,
            'lr_mode': lr_mode,
            'grad_direction': grad_direction,
            'n_epochs': n_epochs,
            'nn_model': nn_model,
            'step_size': step_size,
            'cuda': cuda,
            'cache_middle_result': cache_middle_result
        }
    def getArgs(self):
        return self.args
