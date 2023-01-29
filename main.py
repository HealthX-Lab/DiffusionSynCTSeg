
from src.image_model import Image
from src.datasetscollection_model import DatasetsCollection
from src.trainingchain_model import TrainingChain
from src.testchain_model import TestChain
from configs.base_options import BaseOptions





def main():
    opt = BaseOptions().parse()
    operation = opt.operation
    print(operation,'***')

    if operation == 'Train':
        datasets = DatasetsCollection(opt)
        datasets.over_samples() if opt.oversample else datasets.under_samples()
        paths = datasets.get_data_dics()
        training_chain = TrainingChain(opt,paths)
    elif operation == 'Test' or  operation == 'TestSegMRI' or operation ==  'TestMRI2CT' :
        datasets = DatasetsCollection(opt)
        datasets.paired_samples()
        paths = datasets.get_data_dics()
        test_chain = TestChain(opt, paths)

        print('**')


    # elif operation == 'Train':
    #     training_chain = TrainingChain(attributes['Train'])







if __name__ == "__main__":
    main()