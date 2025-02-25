import configargparse
import argparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    
    parser.add_argument('--inp_dim', 
                        type=int, 
                        default=36, 
                        help='dimension of input')

    parser.add_argument('--batch_size', 
                        type=int, 
                        default=1, 
                        help='number of elements in batch')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=5000, 
                        help='training iterations')

    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-4, 
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--tol', 
                        type=float, 
                        default=1e-2, 
                        help='value of training loss to reach before stopping')

    parser.add_argument('--inp_noise', 
                        type=float, 
                        default=0.01, 
                        help='noise parameter for input')

    parser.add_argument('--act_noise', 
                        type=float, 
                        default=0.01, 
                        help='noise parameter for hidden activity')

    parser.add_argument('--patience', 
                        type=int, 
                        default=50, 
                        help='number of iterations below tolerance before stopping')

    parser.add_argument('--dt', 
                        type=float, 
                        default=10, 
                        help='change in time at each timestep (default: 10)')

    parser.add_argument('--tau', 
                        type=float, 
                        default=100, 
                        help='time constant of neurons (default: 100)')

    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=1e-3, 
                        help='weight decay value (default: 1e-3)')

    parser.add_argument('--seed', 
                        type=int, 
                        default=123456, 
                        help='set manual seed')

    parser.add_argument('--constrained', action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_first', action=argparse.BooleanOptionalAction)

    parser.add_argument('--save_path', 
                        type=str, 
                        default="checkpoints/", 
                        help='path to save network')

    parser.add_argument('--model_specifications_path', 
                        type=str, 
                        default="checkpoints/model_specifications/", 
                        help='path to save network')

    parser.add_argument('--mrnn_config_file', 
                        type=str,
                        default="configurations/mRNN.json",
                        help='path of configuration for mRNN')

    return parser