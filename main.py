

from src.datasetscollection_model import DatasetsCollection
from src.trainingchain_model import TrainingChain
from src.testchain_model import TestChain
from configs.base_options import BaseOptions
from src.image_functions import *





def main():
    opt = BaseOptions().parse()
    operation = opt.operation

    if operation == 'Train':
        datasets = DatasetsCollection(opt)
        datasets.over_samples() if opt.oversample else datasets.under_samples()
        paths = datasets.get_data_dics()
        training_chain = TrainingChain(opt,paths)
        training_chain.build_chain()

    elif operation == 'check_image':
        datasets = DatasetsCollection(opt)
        datasets.over_samples() if opt.oversample else datasets.under_samples()
        paths = datasets.get_data_dics()
        training_chain = TrainingChain(opt, paths)
        val_data = training_chain.get_val_data()
        save_image(opt,val_data)

    elif operation == 'Test':
        datasets = DatasetsCollection(opt)
        datasets.over_samples() if opt.oversample else datasets.under_samples()
        paths = datasets.get_data_dics()
        training_chain = TestChain(opt, paths)
        val_data = training_chain.get_val_data()
        save_image(opt,val_data)


    # elif operation == 'Test' or  operation == 'TestSegMRI' or operation ==  'TestMRI2CT' :
    #     datasets = DatasetsCollection(opt)
    #     datasets.paired_samples()
    #     paths = datasets.get_data_dics()
    #     test_chain = TestChain(opt, paths)



if __name__ == "__main__":
    main()