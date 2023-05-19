import os
import argparse
from pathlib import Path
from __syslog__ import EventlogHandler


@EventlogHandler
def args_m01():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--samples", type=Path, required=True, help = "Path containing the samples")
    ap.add_argument("-s", "--save", type=Path, required=True, help="Backup and saving path")
    ap.add_argument("-b", "--batch", type=int, required=True, help="Batch size")
    ap.add_argument("-m", "--mode", type=int, required=True, choices = [0, 1],  help='Reset Model, Load Model, Evaluate Model. Reset=0, Load=1, Evaluate=2')
    ap.add_argument("--model", type=Path, required=False, help="Path containing the model", default=Path(''))
    
    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")
        
    if not(args.samples.is_file() and args.samples.suffix == '.txt'):
        raise argparse.ArgumentTypeError("\'-samples;--path_samples\' file not found.")
    
    if args.mode == 0:
        if not args.save.is_dir():
            print('Creating save directory: {}'.format(args.save))
            args.save.mkdir(parents=True, exist_ok=True)
        args.model=None
    
    elif args.mode == 1:
        if not(args.model.is_file() and args.model.suffix == '.pt'):
            raise argparse.ArgumentTypeError("for \'--m;--mode\' set to 1 \'--model;--path_model\' must be a valid file.")
            
            
        if not os.path.exists(args.save):
            raise argparse.ArgumentTypeError("the \'--save;--save_model\' is necessary.")
            
    return args
        
def args_m2():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--samples", type=Path, required=True, help = "Path containing the samples")
    ap.add_argument("-s", "--save", type=Path, required=True, help="Backup and saving path")
    ap.add_argument("-b", "--batch", type=int, required=True, help="Batch size")
    ap.add_argument("--model", type=Path, required=True, help="Path containing the model", default=Path(''))
    
    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")
        
    if not args.save.is_dir():
        print('Creating evaluation directory: {}'.format(args.save))
        args.save.mkdir(parents=True, exist_ok=True)
    
    if not(args.model.is_file() and args.model.suffix == '.pt'):
        raise argparse.ArgumentTypeError("for \'--m;--mode\' set to 1 \'--model;--path_model\' must be a valid file.")
    
    return args    
    
if __name__ == "__main__":
    args_m2()
    # args_m01()