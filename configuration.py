class generator_config(object):
    def __init__(self):
        self.emb_dim = 32
        self.num_emb = 5000
        self.hidden_dim = 32
        self.sequence_length = 20
        self.gen_batch_size = 64
        self.start_token = 0
class discriminator_config(object):
    def __init__(self):
        self.sequence_length = 20
        self.num_classes = 2
        self.vocab_size = 5000
        self.dis_embedding_dim = 64
        self.dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.dis_dropout_keep_prob = 0.75
        self.dis_l2_reg_lambda = 0.2
        self.dis_batch_size = 64
        self.dis_learning_rate = 1e-4
class training_config(object):
    def __init__(self):
        self.gen_learning_rate = 0.01
        self.gen_update_time = 1
        self.dis_update_time_adv = 5
        self.dis_update_epoch_adv = 3
        self.dis_update_time_pre = 50
        self.dis_update_epoch_pre = 3
        self.pretrained_epoch_num = 120
        self.rollout_num = 16 
        self.test_per_epoch = 5
        self.batch_size = 64
        self.save_pretrained = 120
        self.grad_clip = 5.0
        self.seed = 88
        self.start_token = 0
        self.total_batch = 200
        self.positive_file = "save/real_data.txt"
        self.negative_file = "save/generator_sample.txt"
        self.eval_file = "save/eval_file.txt"
        self.generated_num = 10000
