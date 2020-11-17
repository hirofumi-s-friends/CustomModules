




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


# One space at the end so the last char of the interface could be "
interface = r"""SCOPESCRIPT PATHIN_Input={input} PARAM_IsSStream={issstream} PARAM_Extract={extract} PARAM_Statement1={statement1} PARAM_Statement2={statement2} PARAM_Statement3={statement3} PARAM_Statement4={statement4} PARAM_Statement5={statement5} PARAM_Statement6={statement6} PARAM_Statement7={statement7} PARAM_Statement8={statement8} PARAM_Statement9={statement9} PARAM_Statement10={statement10} PARAM_IsOutputSStream={isoutputsstream} PARAM_Clustered={clustered} PARAM_Sorting={sorting} PARAM_CSharpCode={csharpcode} PATHOUT_OutputTxt={outputtxt} PATHOUT_OutputSS={outputss}.ss VC=vc://{vc} RETRIES=2 """  


@dsl._component(
    name="scope_arbitrary",
    display_name="""Scope Arbitrary""",
)
def scope_arbitrary(
    input: InputFile(type=(['AnyDirectory', 'AnyFile'])),
    outputtxt: OutputFile(),
    outputss: OutputFile(),
    issstream: str = "true",
    extract: str = "A:string, B:string, C:long",
    statement1: str = "a = SELECT *",
    statement2: str = "b = SELECT *",
    statement3: str = "c = SELECT *",
    statement4: str = "d = SELECT *",
    statement5: str = "e = SELECT *",
    statement6: str = "f = SELECT *",
    statement7: str = "g = SELECT *",
    statement8: str = "h = SELECT *",
    statement9: str = "i = SELECT *",
    statement10: str = "j = SELECT *",
    isoutputsstream: str = "true",
    clustered: str = "//  CLUSTERED BY A, B, C SORTED BY A, B, C",
    sorting: str = "//  ORDERED BY A, B, C",
    csharpcode: str = "//  C# code",
    vc: str = "cosmos09/relevance",
):
    input_file_keys = ["input"]
    port_keys = ["input","outputtxt","outputss"]

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
    ComponentExecutor(scope_arbitrary).execute(sys.argv)
