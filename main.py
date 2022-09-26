import gin
from src.image_model import Image
from src.datasetscollection_model import DatasetsCollection
from src.trainingchain_model import TrainingChain


@gin.configurable("main")
def main(operation, attributes):
    if operation == 'EDA':
        datasets = DatasetsCollection(attributes['EDA'])
        image_obj = Image(datasets.get_paths())
        image_obj.get_images()
        image_obj.show_images()
        image_obj.get_image_size()
    elif operation == 'Train':
        training_chain = TrainingChain(attributes['Train'])







if __name__ == "__main__":
    gin.parse_config_file("./configs/1_config.gin")
    main()