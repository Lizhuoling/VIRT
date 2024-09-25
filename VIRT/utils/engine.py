import logging
import pdb
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import comm

__all__ = ["launch"]


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def launch(main_func, args):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        args (tuple): argparse arguments 
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."

    if int(os.environ['WORLD_SIZE']) > 1:
        dist_url = 'tcp://{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        
        try:
            dist.init_process_group(
                backend="NCCL", init_method=dist_url, world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK'])
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Process group URL: {}".format(args.dist_url))
            raise e
        
        comm.synchronize()

        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        # Setup the local process group (which contains ranks within the same machine)
        assert comm._LOCAL_PROCESS_GROUP is None
        assert int(os.environ['WORLD_SIZE']) % args.num_nodes == 0, "WORLD_SIZE should be divisible by NUM_NODES."
        num_gpus_per_machine = int(os.environ['WORLD_SIZE']) // args.num_nodes
        node_rank = int(os.environ['RANK']) // num_gpus_per_machine
        for i in range(args.num_nodes):
            ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
            pg = dist.new_group(ranks_on_i)
            if i == node_rank:
                comm._LOCAL_PROCESS_GROUP = pg

    main_func(args)