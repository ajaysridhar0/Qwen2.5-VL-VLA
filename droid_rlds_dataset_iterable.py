import torch
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
import time
import json
import logging
from pathlib import Path
import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from enum import Enum
from download_utils import maybe_download

class DroidActionSpace(Enum):
    JOINT_POSITION = 1
    JOINT_VELOCITY = 2

_shuffle_buffer_stats = {
    'lock': __import__('threading').Lock(),
    'buffer_created_count': 0,
    'total_buffer_size': 0,
    'worker_buffers': {}
}

def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

class DroidRldsDatasetStateful(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        dataset_name: str = "droid",
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,   # Aggressive to match threading
        num_parallel_calls: int = -1,   # Aggressive to match threading
        seed: int = 42,
        samples_to_skip: int = 0,
        filter_dict_path: str = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        **kwargs
    ):
        """
        Initializes the dataset configuration. The actual tf.data.Dataset
        will be created in each worker process within the __iter__ method.
        """
        super().__init__()
        # Store all configuration parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.action_chunk_size = action_chunk_size
        self.action_space = action_space
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_calls = num_parallel_calls
        self.seed = seed
        self.samples_to_skip = samples_to_skip
        self.filter_dict_path = filter_dict_path
        # We don't need the GPU-sharding parameters from the original class,
        # as PyTorch's DataLoader and our sharding logic will handle this.
        #     # Initialize filter table in __init__ for efficiency
        self._initialize_filter_table()

    def _initialize_filter_table(self):
        """Initialize the filter table once in __init__ instead of per worker."""
        if self.filter_dict_path is not None:
            cached_filter_dict_path = maybe_download(self.filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            print(f"Initializing filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            print("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )
    
    def __iter__(self):
        """
        This method is called by the DataLoader for each worker process.
        It sets up and yields from a unique, sharded data pipeline.
        """
        # 1. GET WORKER INFO
        # This is the key to multi-process data loading with iterable datasets.
            # 1. GET WORKER INFO
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # print(f"✅ Worker {worker_id}/{num_workers} initializing")

        # 1.1. GET GPU RANK FOR DISTRIBUTED TRAINING
        if torch.distributed.is_initialized():
            gpu_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            gpu_rank = 0
            world_size = 1

        # 2. CONFIGURE TENSORFLOW FOR THE WORKER
        # Each worker needs its own TF configuration to avoid conflicts.
        tf.config.set_visible_devices([], "GPU")

        # 3. BUILD THE TF.DATA PIPELINE (Copied from your original class)
        builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=True, num_parallel_reads=self.num_parallel_reads)

        total_workers = world_size * num_workers
        global_worker_id = gpu_rank * num_workers + worker_id
        
        if total_workers > 1:
            dataset = dataset.shard(total_workers, global_worker_id)

        # 4. SHARD THE DATASET FOR THE CURRENT WORKER
        # This is the most critical step for multi-worker loading.
        # It must be done early, before shuffling and batching.

        # The rest of your data processing pipeline remains the same...
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if self.action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }
        
        # def restructure(traj):
        #     actions = tf.concat(
        #         (
        #             (
        #                 traj["action_dict"]["joint_position"]
        #                 if self.action_space == DroidActionSpace.JOINT_POSITION
        #                 else traj["action_dict"]["joint_velocity"]
        #             ),
        #             traj["action_dict"]["gripper_position"],
        #         ),
        #         axis=-1,
        #     )
        #     exterior_img = tf.cond(
        #         tf.random.uniform(shape=[]) > 0.5,
        #         lambda: traj["observation"]["exterior_image_1_left"],
        #         lambda: traj["observation"]["exterior_image_2_left"],
        #     )
        #     wrist_img = traj["observation"]["wrist_image_left"]
        #     instruction = tf.random.shuffle(
        #         [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
        #     )[0]
        #     return {
        #         "actions": actions,
        #         "observation": {
        #             "image": exterior_img,
        #             "wrist_image": wrist_img,
        #             "joint_position": traj["observation"]["joint_position"],
        #             "gripper_position": traj["observation"]["gripper_position"],
        #         },
        #         "prompt": instruction,
        #     }

        dataset = dataset.traj_map(restructure, self.num_parallel_calls)
        
        # ... [rest of your mapping functions: chunk_actions, filter_idle, decode_images] ...
        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(self.action_chunk_size)[None],
                [traj_len, self.action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, self.action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, self.num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=self.num_parallel_calls)

        # # Filter out frames where actions are idle. Must be done after flattening, as filter should apply per-frame.
        # def filter_idle(traj):
        #     """Filter out chunks with idle actions.
        #     --> we filter if at least first half of chunk does not move.
        #     """
        #     if self.action_space == DroidActionSpace.JOINT_POSITION:
        #         # Compute delta to first position in action chunk
        #         return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
        #     return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2]) > 1e-3)

        # dataset = dataset.filter(filter_idle)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

            #  >> SIMPLE SKIP: Skip samples if specified, otherwise rely on seeding for randomness
        if self.samples_to_skip > 0:
            # Distribute the total skip count among all workers.
            samples_to_skip_per_worker = self.samples_to_skip // total_workers
            remainder = self.samples_to_skip % total_workers
            
            if global_worker_id < remainder:
                samples_to_skip_per_worker += 1
            
            if samples_to_skip_per_worker > 0:
                # Use modulo to handle cases where skip count > dataset size
                # This ensures we don't skip more than the dataset size
                estimated_dataset_size = 20_000_000  # Approximate DROID size after filtering
                safe_skip_count = samples_to_skip_per_worker % estimated_dataset_size
                
                if safe_skip_count != samples_to_skip_per_worker:
                    rank0_print(f"⚠️  Worker {global_worker_id}: Skip count {samples_to_skip_per_worker} > dataset size, using modulo: {safe_skip_count}")
                
                rank0_print(f"🌈⏩ Worker {global_worker_id}/{total_workers}: Skipping {safe_skip_count} samples.")
                dataset = dataset.skip(safe_skip_count)
        else:
            rank0_print(f"🌱 Worker {global_worker_id}: Using seed-based randomness (no skipping)")

        dataset = dataset.repeat()

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, self.num_parallel_calls)

        shuffle_seed = self.seed + global_worker_id if self.seed is not None else None

        # Shuffle, batch, and finalize the pipeline
        dataset = dataset.shuffle(self.shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.with_ram_budget(1)

        # 5. YIELD DATA FROM THE WORKER'S PIPELINE
        # Each worker will now yield batches from its own unique data shard.
        yield from dataset.as_numpy_iterator()

    def __len__(self):
        # As before, this is an approximation.
        return 20_000_000


    
    # def __init__(
    #     self,
    #     data_dir: str,
    #     batch_size: int,
    #     *,  # Force keyword-only arguments
    #     dataset_name: str = "droid",
    #     action_chunk_size: int = 16,
    #     action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
    #     shuffle_buffer_size: int = 250_000,
    #     num_parallel_reads: int = 2,   # Aggressive to match threading
    #     num_parallel_calls: int = 2,   # Aggressive to match threading
    #     data_size: int = None,
    #     # >> STATEFUL: New argument to tell the dataset how many items to skip.
    #     samples_to_skip: int = 0,
    #     # >> REPRODUCIBILITY: Add a seed for deterministic data loading.
    #     seed: int = 42,
    #     # >> JSON FILTER: Path to json file with indices to sample during training
    #     filter_dict_path: str = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
    # ):
    #     """
    #     Initializes the dataset configuration.
    #     """
    #     super().__init__()
    #     # Store all configuration parameters
    #     self.data_dir = data_dir
    #     self.batch_size = batch_size
    #     self.dataset_name = dataset_name
    #     self.action_chunk_size = action_chunk_size
    #     self.action_space = action_space
    #     self.shuffle_buffer_size = shuffle_buffer_size
    #     self.num_parallel_reads = num_parallel_reads
    #     self.num_parallel_calls = num_parallel_calls
    #     self.data_size = data_size
    #     # >> STATEFUL: Store the number of samples to skip.
    #     self.samples_to_skip = samples_to_skip
    #     # >> REPRODUCIBILITY: Store the seed.
    #     self.seed = seed
    #     # >> JSON FILTER: Store the filter dictionary path.
    #     self.filter_dict_path = filter_dict_path
        
    #     # Initialize filter table in __init__ for efficiency
    #     self._initialize_filter_table()

    # def _initialize_filter_table(self):
    #     """Initialize the filter table once in __init__ instead of per worker."""
    #     if self.filter_dict_path is not None:
    #         cached_filter_dict_path = maybe_download(self.filter_dict_path)
    #         with Path(cached_filter_dict_path).open("r") as f:
    #             filter_dict = json.load(f)

    #         print(f"Initializing filter dictionary with {len(filter_dict)} episodes")

    #         keys_tensor = []
    #         values_tensor = []

    #         for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
    #             for start, end in ranges:
    #                 for t in range(start, end):
    #                     frame_key = f"{episode_key}--{t}"
    #                     keys_tensor.append(frame_key)
    #                     values_tensor.append(True)
            
    #         self.filter_table = tf.lookup.StaticHashTable(
    #             tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
    #         )
    #         print("Filter hash table initialized")
    #     else:
    #         self.filter_table = tf.lookup.StaticHashTable(
    #             tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
    #         )

    # def __iter__(self):
    #     """
    #     This method is called by the DataLoader for each worker process.
    #     It sets up and yields from a unique, sharded, and skippable data pipeline.
    #     """
    #     # 1. GET WORKER INFO
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         num_workers = 1
    #         worker_id = 0
    #     else:
    #         num_workers = worker_info.num_workers
    #         worker_id = worker_info.id
    #         # print(f"✅ Worker {worker_id}/{num_workers} initializing")

    #     # 1.1. GET GPU RANK FOR DISTRIBUTED TRAINING
    #     if torch.distributed.is_initialized():
    #         gpu_rank = torch.distributed.get_rank()
    #         world_size = torch.distributed.get_world_size()
    #     else:
    #         gpu_rank = 0
    #         world_size = 1

    #     global_worker_id = gpu_rank * num_workers + worker_id


    #     # 2. CONFIGURE TENSORFLOW FOR THE WORKER
    #     tf.config.set_visible_devices([], "GPU")
        
    #     # ORIGINAL THREADING: Let TensorFlow use default (aggressive) threading
    #     rank0_print(f"🚀 Worker {global_worker_id}: Using TensorFlow default threading for maximum performance")

    #     # >> RANDOM SEEDING: Set a unique seed for each worker to ensure randomness.
    #     # The seed incorporates worker ID and base seed for different data on resume.
    #     if self.seed is not None:
    #         # Each worker gets a unique seed derived from the base seed.
    #         worker_seed = self.seed + global_worker_id
    #         tf.random.set_seed(worker_seed)
    #         print(f"🌱 Worker {global_worker_id}: Set TF random seed to {worker_seed}")

    #     # >> JSON FILTER: Filter table already initialized in __init__
    #     rank0_print(f"Worker {global_worker_id}: Using pre-initialized filter table")

    #     # 3. BUILD THE TF.DATA PIPELINE
    #     builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
    #     dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=True, num_parallel_reads=self.num_parallel_reads)

    #     # 4. SHARD THE DATASET FOR THE CURRENT WORKER AND GPU
    #     total_workers = world_size * num_workers
    #     global_worker_id = gpu_rank * num_workers + worker_id
        
    #     if total_workers > 1:
    #         dataset = dataset.shard(total_workers, global_worker_id)

    #     # The rest of your data processing pipeline remains the same...
    #     dataset = dataset.filter(
    #         lambda traj: tf.strings.regex_full_match(
    #             traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
    #         )
    #     )
        
    #     # NOTE: We perform skipping *before* the repeat() call.
    #     # dataset = dataset.repeat() # This is now handled after skipping logic

    #     # ... [All your mapping and processing functions (restructure, chunk_actions, etc.) go here] ...
    #     # (Code omitted for brevity, it's identical to your original class)
    #     def restructure(traj):
    #         """Reformat observation and action keys, sample language instruction."""
    #         # Important: we use joint *position* action space -- easier to simulate!
    #         actions = tf.concat(
    #             (
    #                 (
    #                     traj["action_dict"]["joint_position"]
    #                     if self.action_space == DroidActionSpace.JOINT_POSITION
    #                     else traj["action_dict"]["joint_velocity"]
    #                 ),
    #                 traj["action_dict"]["gripper_position"],
    #             ),
    #             axis=-1,
    #         )
    #         # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
    #         # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
    #         exterior_img = tf.cond(
    #             tf.random.uniform(shape=[]) > 0.5,
    #             lambda: traj["observation"]["exterior_image_1_left"],
    #             lambda: traj["observation"]["exterior_image_2_left"],
    #         )
    #         wrist_img = traj["observation"]["wrist_image_left"]
    #         # Randomly sample one of the three language instructions
    #         instruction = tf.random.shuffle(
    #             [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
    #         )[0]

    #         traj_len = tf.shape(traj["action"])[0]
    #         indices = tf.as_string(tf.range(traj_len))

    #         # Data filtering:
    #         # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
    #         # and each step's time step index. This will index into the filter hash table, and if it returns true,
    #         # then the frame passes the filter.
    #         step_id = (
    #             traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
    #             + "--"
    #             + traj["traj_metadata"]["episode_metadata"]["file_path"]
    #             + "--"
    #             + indices
    #         )
    #         passes_filter = self.filter_table.lookup(step_id)

    #         return {
    #             "actions": actions,
    #             "observation": {
    #                 "image": exterior_img,
    #                 "wrist_image": wrist_img,
    #                 "joint_position": traj["observation"]["joint_position"],
    #                 "gripper_position": traj["observation"]["gripper_position"],
    #             },
    #             "prompt": instruction,
    #             "step_id": step_id,
    #             "passes_filter": passes_filter,
    #         }

    #     dataset = dataset.traj_map(restructure, self.num_parallel_calls)
        
    #     def chunk_actions(traj):
    #         traj_len = tf.shape(traj["actions"])[0]
    #         action_chunk_indices = tf.broadcast_to(
    #             tf.range(self.action_chunk_size)[None],
    #             [traj_len, self.action_chunk_size],
    #         ) + tf.broadcast_to(
    #             tf.range(traj_len)[:, None],
    #             [traj_len, self.action_chunk_size],
    #         )
    #         action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
    #         traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
    #         return traj

    #     dataset = dataset.traj_map(chunk_actions, self.num_parallel_calls)

    #     # Flatten: map from trajectory dataset to dataset of individual action chunks
    #     dataset = dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    #     # Filter data that doesn't pass the JSON filter
    #     def filter_from_dict(frame):
    #         return frame["passes_filter"]

    #     dataset = dataset.filter(filter_from_dict)

    #     # Remove "passes_filter" key from output
    #     def remove_passes_filter(frame):
    #         frame.pop("passes_filter")
    #         return frame

    #     dataset = dataset.map(remove_passes_filter)

    #     # # Apply the original idle filter as well (keeps both filters)
    #     # def filter_idle(traj):
    #     #     """Filter out chunks with idle actions.
    #     #     --> we filter if at least first half of chunk does not move.
    #     #     """
    #     #     if self.action_space == DroidActionSpace.JOINT_POSITION:
    #     #         # Compute delta to first position in action chunk
    #     #         return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
    #     #     return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2]) > 1e-3)

    #     # dataset = dataset.filter(filter_idle)
        
    #     # >> SIMPLE SKIP: Skip samples if specified, otherwise rely on seeding for randomness
    #     if self.samples_to_skip > 0:
    #         # Distribute the total skip count among all workers.
    #         samples_to_skip_per_worker = self.samples_to_skip // total_workers
    #         remainder = self.samples_to_skip % total_workers
            
    #         if global_worker_id < remainder:
    #             samples_to_skip_per_worker += 1
            
    #         if samples_to_skip_per_worker > 0:
    #             # Use modulo to handle cases where skip count > dataset size
    #             # This ensures we don't skip more than the dataset size
    #             estimated_dataset_size = 20_000_000  # Approximate DROID size after filtering
    #             safe_skip_count = samples_to_skip_per_worker % estimated_dataset_size
                
    #             if safe_skip_count != samples_to_skip_per_worker:
    #                 rank0_print(f"⚠️  Worker {global_worker_id}: Skip count {samples_to_skip_per_worker} > dataset size, using modulo: {safe_skip_count}")
                
    #             rank0_print(f"🌈⏩ Worker {global_worker_id}/{total_workers}: Skipping {safe_skip_count} samples.")
    #             dataset = dataset.skip(safe_skip_count)
    #     else:
    #         rank0_print(f"🌱 Worker {global_worker_id}: Using seed-based randomness (no skipping)")

    #     # >> STREAMING: Repeat the dataset for infinite streaming
    #     dataset = dataset.repeat()
        
    #     # Decode images (this should be after skipping for efficiency)
    #     def decode_images(traj):
    #         traj["observation"]["image"] = tf.io.decode_image(
    #             traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
    #         )
    #         traj["observation"]["wrist_image"] = tf.io.decode_image(
    #             traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
    #         )
    #         return traj
    #     dataset = dataset.frame_map(decode_images, self.num_parallel_calls)

    #     # Data size limit is less relevant for a resuming dataset that repeats,
    #     # but the logic can be kept if needed for single-epoch runs.

    #     # Shuffle, batch, and finalize
    #     # >> RANDOM SEEDING: Use worker-specific seed for shuffle randomness
    #     shuffle_seed = self.seed + global_worker_id if self.seed is not None else None
    #     dataset = dataset.shuffle(self.shuffle_buffer_size, seed=shuffle_seed)
    #     dataset = dataset.batch(self.batch_size)
        
    #     # # OPTIMIZATION: Add prefetching to compensate for conservative threading
    #     # # This creates a buffer that can be filled while training happens
    #     # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    #     dataset = dataset.with_ram_budget(1)

    #     # Note: TensorFlow threading cannot be changed after initialization
    #     # The prefetch buffer helps maintain good training performance
    #     # rank0_print(f"🚀 Worker {global_worker_id}: Shuffle buffer initialized, using prefetch for training performance")

    #     # 5. YIELD DATA FROM THE WORKER'S PIPELINE
    #     yield from dataset.as_numpy_iterator()

    # def __len__(self):
    #     if self.data_size is not None:
    #         return self.data_size
    #     else:
    #         return 20_000_000 # Approximation