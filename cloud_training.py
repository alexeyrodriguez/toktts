import pconfig
import argparse
import subprocess
import os


parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--cloud-config', action = 'store', type = str, help = 'Cloud configuration', required=True)
parser.add_argument('--detach', action='store_true', help = 'Let execution continue even after disconnection')
parser.add_argument('--power-off', action='store_true', help = 'Switch off node after training')

args = parser.parse_args()
cfg = pconfig.load_config(args.cloud_config).gcloud
print(cfg)
print('JEJE', cfg.instance.creation_args)

def gcloud(args, check=True, fake=False):
    print('  Running: '+ ' '.join([cfg.path] + args))
    if not fake:
        res = subprocess.run([cfg.path] + args, check=check)
        res = res.returncode==0
    else:
        res = False

    print()
    return res

def gcloud_start(instance_name):
    return gcloud(['compute', 'instances', 'start', instance_name], check=False)

def gcloud_create_instance(instance_name, creation_args):
    args = [f'--{k}={creation_args[k]}' for k in creation_args]
    return gcloud(['compute', 'instances', 'create', instance_name]+args)

# REMOTE_RUNNER = 'scripts/run_remote_training.sh'
REMOTE_RUNNER = 'prepare_data.py'

def gcloud_remote_training(args, instance_name,
        git_repo_url, git_branch, gcs_model_path,
        wandb_api_key=None, wandb_entity=None, extra_args=[]):
    power_off_arg = 'yes' if args.power_off else 'no'
    command = [
        'bash', os.path.basename(REMOTE_RUNNER), git_repo_url, git_branch, gcs_model_path, power_off_arg
    ]
    command.extend(['--experiment', args.experiment])
    if wandb_api_key:
        command.extend(['--wandb-api-key', wandb_api_key, '--wandb-entity', wandb_entity])
    command.extend(extra_args)
    command = ' '.join(command)
    if args.detach:
        command = f'nohup {command} 2>nohup.err.$$ >nohup.out.$$ ; tail -n 100 -f nohup.out.$$ &'
    return gcloud(['compute', 'ssh', '--command=' + command, instance_name])

if __name__=='__main__':
    instance_name = cfg.instance.name

    print('\nStarting Cloud Training\n')

    print(f'Checking if instance {instance_name} already exists')
    if not gcloud_start(instance_name):
        print(f'Instance {instance_name} not yet available, creating')
        gcloud_create_instance(instance_name, vars(cfg.instance.creation_args))
        print(f'Call the script again to start training, it may fail until setup of the new node is completed')
    else:
        print(f'Using instance: {instance_name}')

        print('Copying remote running script')
        gcloud(['compute', 'scp', REMOTE_RUNNER, f'{instance_name}:'])

        gcloud_remote_training(args, instance_name)