import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

class Environment(CUDAEnvironmentContext):

    def __init__(num_prey = 50, 
                 num_predator = 1, 
                 stage_size = 100.0, 
                 episod_length = 100,
                 max_speed=1.0,
                 max_acceleration=1.0,
                 min_acceleration=-1.0,
                 max_turn=np.pi / 2,
                 min_turn=-np.pi / 2,
                 
                ):
        super().__init__()

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        assert num_prey > 0
        self.num_prey = num_prey

        assert num_predator > 0
        self.num_predator = num_predator
        
        self.num_agents = self.num_prey + self.num_predator