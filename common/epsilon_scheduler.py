import math
class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):

        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
    
    def get_epsilon(self):
        return self.epsilon
