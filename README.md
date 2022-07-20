# Computer Experiment

```r
python3 resnet_data_generate_process.py --algorithm --weighted --lr --low_beta --up_beta --momentum
```


`parser.add_argument("--algorithm", "-a", type=str, nargs='?', default="Adam", choices=["Adam", "Adamax", "NAdam", "SGD", "RMSprop"] ,help="Optimizer")
parser.add_argument("--weighted", "-w", type=bool, nargs='?', default=True, help="Use weighted loss function")
parser.add_argument("--lr", "-l", type=float, nargs='?', default=0.001, help="Learning rate")
parser.add_argument("--low_beta", "-lb", type=float, nargs='?', default=0.9, help="Decay rate (lower)")
parser.add_argument("--up_beta", "-ub", type=float, nargs='?', default=0.999, help="Decay rate (upper)")
parser.add_argument("--momentum", "-m", type=float, nargs='?', default=0.9, help="Momentum in SGD")
`
 
