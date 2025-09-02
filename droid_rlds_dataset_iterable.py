import torch
from torch.utils.data import IterableDataset, DataLoader

# It's good practice to import these within the methods that use them,
# especially in a worker-based setup, but for clarity we'll import them here.
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds
from enum import Enum # Assuming you have this for DroidActionSpace

# Assuming this Enum exists from your original code
class DroidActionSpace(Enum):
    JOINT_POSITION = 1
    JOINT_VELOCITY = 2


class DroidRldsDatasetIterable(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        dataset_name: str = "droid",
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
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
        
        # We don't need the GPU-sharding parameters from the original class,
        # as PyTorch's DataLoader and our sharding logic will handle this.
    
    def __iter__(self):
        """
        This method is called by the DataLoader for each worker process.
        It sets up and yields from a unique, sharded data pipeline.
        """
        # 1. GET WORKER INFO
        # This is the key to multi-process data loading with iterable datasets.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Main process
            num_workers = 1
            worker_id = 0
        else:
            # A worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # 2. CONFIGURE TENSORFLOW FOR THE WORKER
        # Each worker needs its own TF configuration to avoid conflicts.
        tf.config.set_visible_devices([], "GPU")

        # 3. BUILD THE TF.DATA PIPELINE (Copied from your original class)
        builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=True, num_parallel_reads=self.num_parallel_reads)

        # 4. SHARD THE DATASET FOR THE CURRENT WORKER
        # This is the most critical step for multi-worker loading.
        # It must be done early, before shuffling and batching.
        if num_workers > 1:
            dataset = dataset.shard(num_workers, worker_id)

        # The rest of your data processing pipeline remains the same...
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        dataset = dataset.repeat()
        
        def restructure(traj):
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
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]
            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
            }

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

        # Filter out frames where actions are idle. Must be done after flattening, as filter should apply per-frame.
        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if self.action_space == DroidActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: self.action_chunk_size // 2]) > 1e-3)

        dataset = dataset.filter(filter_idle)

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

        # Shuffle, batch, and finalize the pipeline
        dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.with_ram_budget(1)

        # 5. YIELD DATA FROM THE WORKER'S PIPELINE
        # Each worker will now yield batches from its own unique data shard.
        yield from dataset.as_numpy_iterator()

    def __len__(self):
        # As before, this is an approximation.
        return 20_000_000

