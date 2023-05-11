## Login to the cluster

For running gym environments that require a screen as in our custom gym environments the -X flag for ssh command is required.

## Navigation commands


```
cd mujoco-eleanor

```

```
cd mujoco-eleanor/singularity

```

## Build on the cluster

```
sudo singularity build singularity/eleanor.sif singularity/image.def

```


## Run RL

Explanation:

```
sbatch run_script.sh env_name

```

Example:

```
sudo sbatch run_script.sh PegInHole-rand_events_visual_servoing_guiding2

```


## Print progress out 

```
tail -n 20 slurm-xxx.out

```



