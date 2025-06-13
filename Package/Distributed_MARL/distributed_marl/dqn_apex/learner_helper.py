from collections import deque
import ray 
import time 
import zmq
import pyarrow as pa
import pdb

@ray.remote
class LearnerHelper:
    """
    help learner better update batch
    """
    def __init__(self, config):
        self.replay_buffer = deque(maxlen=100)
        self.priority_buffer = deque(maxlen=100)
        self.pairport_tolearner = config['pairport_tolearner']
        self.pairport_tobuffer = config['pairport_tobuffer']
        self.initialize_sockets()
        self.first_replay_send = False

    def initialize_sockets(self):

        # For receive priority data and send new replay data to learner
        context = zmq.Context()
        self.learner_socket = context.socket(zmq.PAIR)
        self.learner_socket.connect(f"tcp://127.0.0.1:{self.pairport_tolearner}")
        # For receiving batch from, sending new priorities to Buffer 
        context = zmq.Context()
        self.buffer_socket = context.socket(zmq.PAIR)
        self.buffer_socket.bind(f"tcp://127.0.0.1:{self.pairport_tobuffer}")

    def send_replay_data(self):

        if len(self.replay_buffer) > 0 :
            replay_data = self.replay_buffer.pop()
            replay_data_id = pa.serialize(replay_data).to_buffer()
            self.learner_socket.send(replay_data_id)
            self.first_replay_send = True
        else: 
            return
        
    def send_priority(self):

        if len(self.priority_buffer) > 0:
            priority = self.priority_buffer.pop()
            priority_id = pa.serialize(priority).to_buffer()
            self.buffer_socket.send(priority_id)
        else: 
            return 

    def receive_replay_data(self):
        
        replay_data_id = False
        try:
            replay_data_id = self.buffer_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if replay_data_id:
            replay_data = pa.deserialize(replay_data_id)
            self.replay_buffer.append(replay_data)
            
    def receive_priority(self):

        priority_id = False
        try:
            priority_id = self.learner_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass 
        if priority_id:
            priority = pa.deserialize(priority_id)
            self.priority_buffer.append(priority)
            self.send_replay_data()

    def run(self):
        try:
            time.sleep(3)
            print('learner helper starts running')
            # s = time.time()
            while True:
                self.receive_replay_data()
                if not self.first_replay_send:
                    self.send_replay_data()
                if len(self.replay_buffer) > 0:
                    self.receive_priority()
                self.send_priority()
                # if (time.time() - s)>1: 
                #     print('replay_buff:', len(self.replay_buffer))
                #     print('prior_buff:', len(self.priority_buffer))
                #     s = time.time()
        except KeyboardInterrupt:
            import sys 
            sys.exit()