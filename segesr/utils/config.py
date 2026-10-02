import argparse
import yaml

def parse_args_with_config(parser, input_args=None):
    """
    Parses arguments with optional YAML config support.

    Values from `--config` are applied as parser defaults, so any flag passed explicitly on
    the command line still overrides them. Unknown keys in the YAML file raise an error,
    so typos never get silently ignored.
    """

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(input_args)

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML file whose keys are argument names. Command line flags override it.",
    )

    if config_args.config is not None:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f) or {}

        if not isinstance(config, dict):
            raise ValueError(f"Config file '{config_args.config}' must contain a mapping of argument names to values.")

        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file '{config_args.config}': {unknown_keys}")

        for action in parser._actions:
            if action.dest not in config:
                continue

            # Required arguments provided by the config must not be required on the CLI anymore
            action.required = False

            # argparse does not apply `type` to defaults, and PyYAML reads e.g. `2e-7` as a string
            value = config[action.dest]
            if action.type is not None and value is not None:
                if isinstance(value, list):
                    config[action.dest] = [action.type(v) if isinstance(v, str) else v for v in value]
                elif isinstance(value, str):
                    config[action.dest] = action.type(value)

            if action.nargs in ("+", "*") and value is not None and not isinstance(config[action.dest], list):
                config[action.dest] = [config[action.dest]]

        parser.set_defaults(**config)

    return parser.parse_args(input_args)
