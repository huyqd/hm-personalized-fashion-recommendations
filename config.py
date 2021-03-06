class Config:
    TRAIN_SIZE = 0.9
    TEST_SIZE = 1 - TRAIN_SIZE
    RANDOM_SEED = 42
    N_RECOMMENDATIONS = 12
    TRAIN_NEGATIVE_SAMPLES = 4
    VAL_NEGATIVE_SAMPLES = TEST_NEGATIVE_SAMPLES = 12
