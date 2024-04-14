
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or opt.dataset_mode == 'yh' or opt.dataset_mode == 'yh_seg')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'cycle_seg':
        assert(opt.dataset_mode == 'yh_seg' or opt.dataset_mode == 'yh_seg_spleen')
        from .cycle_seg_model import CycleSEGModel
        model = CycleSEGModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'yh_seg')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'test_seg':
        assert(opt.dataset_mode == 'yh_seg')
        from .test_seg_model import TestSegModel
        model = TestSegModel()
    elif opt.model == 'mri_seg':
        assert(opt.dataset_mode == 'yh_seg')
        from .mri_segmentation import MRISEGModel
        model = MRISEGModel()
    elif opt.model == 'mri_test':
        assert(opt.dataset_mode == 'yh_seg')
        from .test_mri_model import MRITestModel
        model = MRITestModel()
    elif opt.model == 'finetune':
        assert(opt.dataset_mode == 'yh_seg')

    elif opt.model == 'TestGANModel':
        assert (opt.dataset_mode == 'yh_seg')
        from .test_cycle_gan import TestGANModel
        model = TestGANModel()


    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
