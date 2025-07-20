import os
import copy
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# 𝜇1: BitRate Reward;   𝜇2: Smooth Penalty;   𝜇3: Buffer Penalty;
beta = 6
QoE_Param_Type = {
    "livestreams": {"𝜇1": 1, "𝜇2": 1, "𝜇3": beta},
    "documentaries": {"𝜇1": 1, "𝜇2": beta, "𝜇3": 1},
    "news": {"𝜇1": beta, "𝜇2": 1, "𝜇3": 1},
    "normal": {"𝜇1": 1, "𝜇2": 0.5, "𝜇3": 4.3},
}


LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
M_IN_K = 1000.0
NOISE_HIGH = 1.1
SMOOTH_PENALTY = 0.5
BITRATE_LEVELS = 6

REBUF_PENALTY = 4.3  # 4.3  # 1 sec rebuffering -> 3 Mbps
DEFAULT_QUALITY = 1  # default video quality without agent


BUFFER_NORM_FACTOR = 10.0

MILLISECONDS_IN_SECOND = 1000.0
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit


class VideoStreaming(gym.Env):
    def __init__(self, trace_name, bandwidth_type, qoe_type, seed):
        super(VideoStreaming, self).__init__()
        self.seed(seed)

        assert trace_name in ["fcc", "hsdpa", "oboe"], f"Invalid trace name: {trace_name}"
        assert bandwidth_type in ["high", "low", "hybrid"], f"Invalid bandwidth type: {bandwidth_type}"

        self.VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
        self.TOTAL_VIDEO_CHUNCK = 48
        

        self.time_traces, self.bandwidth_traces = self._load_bandwidth_trace(trace_name, bandwidth_type)
        self.trace_index = np.random.randint(len(self.time_traces))
        self.current_trace_times = self.time_traces[self.trace_index]
        self.current_bandwidth = self.bandwidth_traces[self.trace_index]
        self.bandwidth_ptr = np.random.randint(1, len(self.current_bandwidth))
        self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

        self.video_chunk_sizes = self._load_video_sizes_by_bitrate()
        self.last_select_bitrate = 0

        self.chunk_index = 0
        self.num_state_features = 6
        self.state_history_length = 8

        self.action_space = gym.spaces.Discrete(BITRATE_LEVELS)
        self.observation_space = self.observation_space = spaces.Dict(
            {
                "delay_ms": spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "sleep_time_ms": spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "buffer_size_ms": spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "rebuffer_ms": spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "selected_video_chunk_size_bytes": spaces.Box(low=0, high=1e8, shape=(), dtype=np.int32),
                "is_done_bool": spaces.Discrete(2),  # 0 or 1
                "remain_chunk": spaces.Box(low=0, high=self.TOTAL_VIDEO_CHUNCK, shape=(), dtype=np.int32),
                "next_video_chunk_sizes": spaces.Box(low=0, high=1e8, shape=(BITRATE_LEVELS,), dtype=np.int32),
            }
        )

    def seed(self, seed_num):
        self.seed_num = seed_num
        if seed_num is not None:
            np.random.seed(seed_num)
            random.seed(seed_num)
            if hasattr(self, "action_space"):
                self.action_space.seed(seed_num)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.time_stamp = 0
        self.client_buffer_size = 0  # ms
        self.video_chunk_cnt = 0
        self.state = np.zeros((self.num_state_features, self.state_history_length))
        self.last_select_bitrate = 0
        state_dict = self._get_video_chunk(self.last_select_bitrate)
        assert state_dict["remain_chunk"] < self.TOTAL_VIDEO_CHUNCK, "Video Chunk Remain Error!"
        return copy.deepcopy(state_dict), {}

    def step(self, action):
        bitrate = int(action)
        state_dict = self._get_video_chunk(bitrate)
        bitrate_reward = math.log(self.VIDEO_BIT_RATE[bitrate] / M_IN_K + 0.7) - (0.01 / (self.VIDEO_BIT_RATE[bitrate] / M_IN_K))  # range in [0, 1.6]
        rebuffer_time_reward = REBUF_PENALTY * state_dict["rebuffer_ms"] / MILLISECONDS_IN_SECOND
        smooth_penalty_reward = SMOOTH_PENALTY * np.abs(self.VIDEO_BIT_RATE[bitrate] - self.VIDEO_BIT_RATE[self.last_select_bitrate]) / M_IN_K
        reward = bitrate_reward - rebuffer_time_reward - smooth_penalty_reward
        
        # Update State Info
        self.last_select_bitrate = bitrate
        terminated = bool(state_dict["is_done_bool"])
        truncated = bool(False)
        return (
            copy.deepcopy(state_dict),
            float(reward),
            terminated,
            truncated,
            {"bitrate_reward": bitrate_reward, "rebuffer_time_reward": -rebuffer_time_reward, "smooth_penalty_reward": -smooth_penalty_reward},
        )

    def _get_video_chunk(self, quality):
        assert quality >= 0, "Video Quality Must be Greater 0!"
        assert quality < BITRATE_LEVELS, "Video Quality Must be Less than BITRATE_LEVELS!"
        selected_chunk_size = self.video_chunk_sizes[quality][self.video_chunk_cnt]

        ## --------- Setting Parameters --------- ##
        BITS_IN_BYTE = 8.0
        B_IN_MB = 1000000.0
        PACKET_PAYLOAD_PORTION = 0.95

        ## --------- Process Video Chunk by Network --------- ##
        delay = 0.0  # ms
        video_chunk_have_processed = 0  # in bytes
        while True:

            throughput = self.current_bandwidth[self.bandwidth_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.current_trace_times[self.bandwidth_ptr] - self.last_bandwidth_time
            assert duration > 0, "duration time must > 0!"
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_have_processed + packet_payload > selected_chunk_size:
                fractional_time = (selected_chunk_size - video_chunk_have_processed) / throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time  # current delay (s)
                self.last_bandwidth_time += fractional_time
                assert self.last_bandwidth_time <= self.current_trace_times[self.bandwidth_ptr], "bw timestamp must less than next ptr timestamp!"
                break

            video_chunk_have_processed += packet_payload
            delay += duration
            self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr]
            self.bandwidth_ptr += 1
            if self.bandwidth_ptr >= len(self.current_bandwidth):
                self.bandwidth_ptr = 1
                self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

        ## --------- Add Noise For Link --------- ##
        delay *= MILLISECONDS_IN_SECOND  # delay (ms)
        delay += LINK_RTT
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)  # add a multiplicative noise to the delay

        ## ---------- Process Buffer Time And Buffer Size --------- ##
        wait_rebuf_time = np.maximum(delay - self.client_buffer_size, 0.0)  # wait rebuffer time, ms
        self.client_buffer_size = np.maximum(self.client_buffer_size - delay, 0.0)  # update the buffer
        VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
        self.client_buffer_size += VIDEO_CHUNCK_LEN  # add in the new chunk

        ## --------- Sleep If Buffer Gets Too Large --------- ##
        sleep_time = 0  # ms
        if self.client_buffer_size > BUFFER_THRESH:  # buffer > 60000ms ---> sleep
            # We need to skip some network bandwidth here but not add up the delay

            ## --------- Compute Sleep Time --------- ##
            drain_buffer_time = self.client_buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            self.client_buffer_size -= sleep_time
            while True:  # Network Move Forward
                sleep_duration = self.current_trace_times[self.bandwidth_ptr] - self.last_bandwidth_time
                if sleep_duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_bandwidth_time += sleep_time / MILLISECONDS_IN_SECOND
                    break

                sleep_time -= sleep_duration * MILLISECONDS_IN_SECOND
                self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr]
                self.bandwidth_ptr += 1

                if self.bandwidth_ptr >= len(self.current_bandwidth):
                    # loop back in the beginning, trace file starts with time 0.
                    self.bandwidth_ptr = 1
                    self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

        ## --------- Update Video Chunk Information --------- ##
        self.video_chunk_cnt += 1
        video_chunk_remain = self.TOTAL_VIDEO_CHUNCK - self.video_chunk_cnt
        end_of_video = False
        if self.video_chunk_cnt >= self.TOTAL_VIDEO_CHUNCK:
            end_of_video = True

            ## --------- Reset Buffer And Video Chunk Cnt --------- ##
            self.client_buffer_size = 0
            self.video_chunk_cnt = 0

            ## ---------Reset Select network bandwidth --------- ##
            self.trace_index = np.random.randint(len(self.time_traces))

            self.current_trace_times = self.time_traces[self.trace_index]
            self.current_bandwidth = self.bandwidth_traces[self.trace_index]
            self.bandwidth_ptr = np.random.randint(1, len(self.current_bandwidth))
            self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

        next_video_chunk_sizes = []
        for level in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_chunk_sizes[level][self.video_chunk_cnt])
        return {
            "delay_ms": np.array(delay, dtype=np.float32),
            "sleep_time_ms": np.array(sleep_time, dtype=np.float32),
            "buffer_size_ms": np.array(self.client_buffer_size, dtype=np.float32),
            "rebuffer_ms": np.array(wait_rebuf_time, dtype=np.float32),
            "selected_video_chunk_size_bytes": np.array(selected_chunk_size, dtype=np.int32),
            "is_done_bool": end_of_video,
            "remain_chunk": np.array(video_chunk_remain, np.int32),
            "next_video_chunk_sizes": np.array(next_video_chunk_sizes, dtype=np.int32),
        }

    def _load_bandwidth_trace(self, trace_folder_name, bandwidth_category):
        """
        Load bandwidth trace data from the specified folder.

        Args:
            trace_folder_name (str): Name of the folder containing trace files.
            bandwidth_category (str): Bandwidth category: 'low', 'high', or 'hybrid'.

        Returns:
            list[list[float]]: List of time sequences.
            list[list[float]]: List of bandwidth sequences (each value multiplied by 2).
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        trace_dir = os.path.abspath(os.path.join(base_path, "..", "trace", trace_folder_name))

        time_sequences = []
        bandwidth_sequences = []

        for file_name in os.listdir(trace_dir):
            file_path = os.path.join(trace_dir, file_name)
            times = []
            bandwidths = []

            with open(file_path, "rb") as file:
                for line in file:
                    parts = line.split()
                    times.append(float(parts[0]))
                    bandwidths.append(float(parts[1]) * 2)

            avg_bandwidth = np.mean(bandwidths)

            if bandwidth_category == "low" and avg_bandwidth < 2.0:
                time_sequences.append(times)
                bandwidth_sequences.append(bandwidths)
            elif bandwidth_category == "high" and avg_bandwidth > 3.0:
                time_sequences.append(times)
                bandwidth_sequences.append(bandwidths)
            elif bandwidth_category == "hybrid":
                time_sequences.append(times)
                bandwidth_sequences.append(bandwidths)
            else:
                continue  # Skip file if it doesn't meet the condition

        return time_sequences, bandwidth_sequences

    def _load_video_sizes_by_bitrate(self):
        """
        Load video chunk sizes for each bitrate level.

        Returns:
            dict[int, list[int]]: A dictionary where each key is a bitrate level
                                and the value is a list of video chunk sizes (in bytes).
        """
        video_sizes = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        size_file_prefix = os.path.abspath(os.path.join(base_path, "..", "videosize", "ori", "video_size_"))

        for bitrate_level in range(BITRATE_LEVELS):
            video_sizes[bitrate_level] = []
            size_file_path = f"{size_file_prefix}{bitrate_level}"

            with open(size_file_path, "r") as file:
                for line in file:
                    # Assume the first column in each line is the chunk size in bytes
                    chunk_size = int(line.split()[0])
                    video_sizes[bitrate_level].append(chunk_size)

        return video_sizes
