from models import cnn_model
# from models import gan_model


def create_model(opts):
    if opts.model_type == 'model_cnn':
        model = cnn_model.CNNModel(opts)

    # elif opts.model_type == 'model_gan':
    #     model = gan_model.GANModel(opts)

    else:
        raise NotImplementedError

    return model
