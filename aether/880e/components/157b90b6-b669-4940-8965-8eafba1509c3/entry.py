




import sys
import subprocess
import re
import shlex
from pathlib import Path
from enum import Enum
from azure.ml.component import dsl
from azure.ml.component.dsl._component import ComponentExecutor, InputPath, OutputPath, InputFile, OutputFile


optional_pattern = re.compile(r'(\[[\s\S]*?\])')
variable_pattern = re.compile(r'(\{[\s\S]*?\})')


def get_optional_mapping(interface):
    """Get the mapping from an optional variable to the corresponding command line in []."""
    return {variable_pattern.search(item).group(0): item[1:-1] for item in re.findall(optional_pattern, interface)}


def get_input_file(path: str):
    """Get the file of the input to workaround that windows compute will output a file as a directory."""
    path = Path(path)
    if path.is_dir():
        files = list(path.iterdir())
        return str(files[0])
    elif path.is_file():
        return path
    else:
        raise FileNotFoundError("Input file cannot be found: %s" % path)


class Enum_outputisstructuredstream(Enum):
    true = 'true'
    false = 'false'


# One space at the end so the last char of the interface could be "
interface = r"""SCOPESCRIPT PATHIN_Input={input} PATHOUT_Output={output}.ss PATHOUT_Counts={counts}.ss PARAM_Preamble={preamble} PARAM_ScopeCode={scopecode} PARAM_OutputIsStructuredStream={outputisstructuredstream} PARAM_OutputSpecification={outputspecification} PARAM_CSharpCode={csharpcode} VC=vc://{vc} RETRIES=2 """  


@dsl._component(
    name="sstreamprocessor_with_counts",
    display_name="""SStreamProcessor with counts""",
)
def sstreamprocessor_with_counts(
    input: InputFile(type=(['AnyDirectory', 'AnyFile'])),
    output: OutputFile(),
    counts: OutputFile(),
    preamble: str = "//",
    scopecode: str = "SELECT * FROM Input;",
    outputisstructuredstream: Enum_outputisstructuredstream = Enum_outputisstructuredstream.true,
    outputspecification: str = "//",
    csharpcode: str = "//",
    vc: str = "cosmos08/AWE.Prod",
):
    input_file_keys = ["input"]
    port_keys = ["input","output","counts"]

    func_args = {k: v for k, v in locals().items()}
    for k in port_keys:
        if func_args[k]:
            func_args[k] = str(Path(func_args[k]).absolute().resolve())  # Convert path format to align the OS.
    
    for k in input_file_keys:
        if func_args[k]:
            func_args[k] = get_input_file(func_args[k])  # Convert input directory to an input file to workaround.

    args = interface.strip()
    mapping = get_optional_mapping(args)

    for optional in mapping.values():
        args = args.replace('[%s]' % optional, optional)  # Remove optional tag "[]" in the interface.

    for k, v in func_args.items():
        goal = '{%s}' % k
        if v is None:
            args = args.replace(mapping.get(goal, goal), '')  # Remove all optional values which is None.

    # Replace every argument with the real variable value, note that for enum values, we need to use v.value to get proper str value.
    func_args = {k: v.value if isinstance(v, Enum) else str(v) for k, v in func_args.items()}
    args = [arg.format(**func_args) for arg in shlex.split(args)]
    print(args)
    # subprocess.run(args, check=True)


if __name__ == '__main__':
    ComponentExecutor(sstreamprocessor_with_counts).execute(sys.argv)