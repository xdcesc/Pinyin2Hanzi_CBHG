class Hyperparams:
    '''Hyper parameters'''
    isqwerty = True # If False, 10 keyboard layout is assumed.
         
    # model
    embed_size = 300 # alias = E
    hidden_size = 128
    encoder_num_banks = 16
    num_highwaynet_blocks = 4
    maxlen = 50 # maximum number of a pinyin sentence
    minlen = 10 # minimum number of a pinyin sentence
    norm_type = "bn" # Either "bn", "ln", "ins", or None
    dropout_rate = 0.5

    checkpoint_path = 'lmmodel_save'
    # training scheme
    lr = 0.001
    logdir = "log/model" if isqwerty  is True else "log/model"
    #batch_size = 64
    #num_epochs = 20
    batch_size = 200
    num_epochs = 30