from rl.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion
from rl.tasks.bin_packing.bin_task_packing import BinTaskPacking
from rl.tasks.book_shelving.book_shelving_task import BookShelvingTask
from rl.tasks.amazon_drawer.drawer_task_pulling import DrawerTaskPulling

isaacgym_task_map = {
    'TacSLTaskInsertion': TacSLTaskInsertion,
    'BinTaskPacking': BinTaskPacking,
    'BookShelvingTask': BookShelvingTask,
    'DrawerTaskPulling': DrawerTaskPulling,
}