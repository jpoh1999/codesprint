from model.model import Model
from libs.utils import *
from libs.constants import *

base_model = "base_model"
model_config = load_config_file(CONFIG_FILE_PATH)["model"]
print(model_config)
logger = configure_logger(base_model, f"{base_model}.log")
file_list = [f for f in os.listdir(load_config_file(CONFIG_FILE_PATH)["input"][1]) if f.endswith('.csv')]

model = Model(logger=logger,
              model_config=model_config,
              height_factor=[1,100,1000,10000,100000,1000000])

model.tune_model(file_list, "input/slot_profiles", "model/tune")

