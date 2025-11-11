from .args import CliArgs, plugins

def main():

    args = CliArgs.parse_args()

    assert args.command
    
    pl_info = plugins.get(args.command)

    if pl_info is None:
        raise ValueError(f"Unknown command: {args.command}")

    pl_info["processor"](args)

if __name__ == "__main__":
    main()
