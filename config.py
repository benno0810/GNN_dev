class InitLearningRate():
    def __init__(self):
        self.init_lr = {
            'karate_34': 1e-5,
            'jazz_198': 5e-5,
            'lesmis_77': 5e-4,
            'copperfield_112': 5e-3,
            'celeganmetabolic_453': 5e-7,
            'celegansneural_297': 1e-5,
            'email_1133': 5e-4,
            'USAir97_332': 1e-5,
            'USairports_1858': 1e-3,
            'polbooks_105':1e-5
        }


    def get_init_lr(self,dataset):
        return self.init_lr[dataset]