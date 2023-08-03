from . import cardiacspect_dataset

def get_datasets(opts):
    if opts.dataset == 'CardiacSPECT':
        trainset = cardiacspect_dataset.CardiacSPECT_Train(opts)
        validset = cardiacspect_dataset.CardiacSPECT_Valid(opts)
        testset  = cardiacspect_dataset.CardiacSPECT_Test(opts)

    elif opts.dataset == 'XXX':
        raise NotImplementedError

    return trainset, validset, testset
