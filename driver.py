from util._util import get_parser_options
from src.trainer_class import *


opt = get_parser_options()

trainer = ModReduce_trainer(opt)

trainer.train()