import os
import shutil
import subprocess
import sys

import yaml
from absl import app, flags

from pdb_file_preprocess import preprosess_pdb

FLAGS = flags.FLAGS

# task args
flags.DEFINE_enum('task_type', 'monomer', ['monomer', 'mini_binder', 'vhh', 'scfv'], ' task type')
flags.DEFINE_string('outdir', None, 'out put dir for design tasks')
flags.DEFINE_string('config', None, 'configration files folder (containing base.yaml and task_type.yaml)')
flags.DEFINE_integer('random_seed', 1, 'random seed value')
flags.DEFINE_string('json_path', None, 'json file path for monomer task')
flags.DEFINE_integer('design_length',62, 'the lenght for current design task, only for monomer and mini_binder')
flags.DEFINE_string('framework_cif_path', '', 'Optional path to a framework-only mmCIF for binder nanobody design. ')
flags.DEFINE_string('pdb_path', None, 'pdb file path for mini_biner and vhh tasks')
flags.DEFINE_string('hotspot_indices', '', 'Optional hotspot_indices for mini_biner and vhh tasks')
flags.DEFINE_string('target_full_seq', '', 'Optional path to target pdb full seq .csv file, for mini_biner and vhh tasks')


def load_yaml(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data else {}
    return {}

def main(argv):
    # Verify the required parameters 
    # running like: python run_task.py --task_type=mini_binder --config=./config --outdir=./results --design_length=100
    if not FLAGS.config or not FLAGS.outdir:
        print("Error: must provide config folder --config and out_put folder --outdir")
        sys.exit(1)

    # 1. Obtaining the Configuration Directory and Task Type
    config_dir = FLAGS.config
    task_type = getattr(FLAGS, 'task_type')

    # 2. Hierarchical loading and merging of configurations.
    # first loading base.yaml, then loading task yaml (e.g. monomer.yaml)
    final_params = load_yaml(os.path.join(config_dir, 'base.yaml'))
    final_params.update(load_yaml(os.path.join(config_dir, f'{task_type}.yaml')))

    # mini_binder, vhh and scfv tasks need to preprocess pdb files to generate json files
    if FLAGS.task_type in ['mini_binder', 'vhh', 'scfv']:
        if not FLAGS.pdb_path:
            print("Error: mini_binder, vhh and scfv task must provide pdb file --pdb_path")
            sys.exit(1)

        # preprocess pdb file
        json_path, hotspot_indices = preprosess_pdb(
            input_pdb_file=FLAGS.pdb_path,
            input_hotspot_string=FLAGS.hotspot_indices,
            output_root_dir=FLAGS.outdir,
            target_full_seq_csvfile=FLAGS.target_full_seq
        )
        # update json_path and hotspot indices
        final_params['hotspot_indices'] = hotspot_indices
        final_params['json_path'] = json_path

    if FLAGS.task_type in ['monomer']:
        # need provide json_path
        if not FLAGS.json_path:
            print("Error: monomer task must provide a json file --json_path")
            sys.exit(1)
        final_params['json_path'] = FLAGS.json_path


    # 3. Inject the args values passed by the CLI into the params dictionary
    final_params['output_dir'] = FLAGS.outdir
    # final_params['json_path'] = FLAGS.json_path
    final_params['random_seed'] = FLAGS.random_seed
    if FLAGS.task_type in ['monomer', 'mini_binder']:
    # updated desing_length only for monomer and mini_binder task
        final_params['design_length'] = FLAGS.design_length
    if FLAGS.task_type in ['vhh', 'scfv']:
        final_params['framework_cif_path'] = FLAGS.framework_cif_path

    # 3.5 Inject uppercase yaml keys as environment variables for the subprocess
    #     e.g. TRIANGLE_MULTIPLICATIVE, TRIANGLE_ATTENTION (cuequivariance acceleration)
    env_overrides = {}
    for key in list(final_params.keys()):
        if key.isupper():
            env_overrides[key] = str(final_params.pop(key))

    # 4. Build the underlying execution command
    # Use current __file__ to get base_dir, i.e. repo base_dir
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_bin = os.getenv("PYTHON_BIN", "python")
    backend_script = os.path.join(base_dir, "src/run_torchcraft.py")



    cmd = [python_bin, backend_script]
    
    # Converts all configuration items in the dictionary to the --key=value format
    for key, value in final_params.items():
        # If the value is a list or a special string, keep its original format
        cmd.append(f"--{key}={value}")

    print(f"--- Starting the torchcraft task... [task_type: {task_type}] ---")
    print(f"Configuration file directory: {config_dir}")
    print(f"Final output path: {FLAGS.outdir}")
    print(f"The JSON file is: {final_params['json_path']}")
    print(f"Current random_seed value: {final_params['random_seed']}")
    print(f"Executing: {backend_script}")
    
    # Execute the script
    env = os.environ.copy()
    env.update(env_overrides)
    if env_overrides:
        print(f"Environment overrides: {env_overrides}")
    subprocess.run(cmd, check=True, env=env)

    if FLAGS.task_type in ['mini_binder', 'vhh', 'scfv']:
        output_cif_dir = os.path.join(FLAGS.outdir, "target_cif")
        output_json_dir = os.path.join(FLAGS.outdir, "target_json")
        # remove the dir and its contents
        if os.path.exists(output_cif_dir):
            shutil.rmtree(output_cif_dir)
        if os.path.exists(output_json_dir):
            shutil.rmtree(output_json_dir)


if __name__ == '__main__':
    app.run(main)