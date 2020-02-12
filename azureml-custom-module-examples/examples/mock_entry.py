import sys
import re
import argparse

input_re = re.compile(r'{{(input_\d+)}}')
output_re = re.compile(r'{{(output_\d+)}}')
param_re = re.compile(r'{{(param_\d+)}}')


def generate_args(interface, sys_args):
    replacements = []
    parser = argparse.ArgumentParser()
    for arg_re in [input_re, output_re, param_re]:
        for item in re.findall(arg_re, interface):
            replacements.append(item)
            parser.add_argument("--" + item.replace('_', '-'), default='')

    args, _ = parser.parse_known_args(sys_args)
    result = interface
    for replacement in replacements:
        result = result.replace('{{%s}}' % replacement, getattr(args, replacement))
    return result


def parse_args(sys_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface')
    args, _ = parser.parse_known_args(sys_args)
    interface_string = args.interface.strip('\\n')
    return generate_args(interface_string, sys_args)


if __name__ == '__main__':
    print("This is a mock entry, will try generate the real args.")
    print()
    print(parse_args(sys.argv))
    print()
    print('Original args', len(sys.argv) - 1, "arguments:")
    print('\n'.join(sys.argv[1:]))
