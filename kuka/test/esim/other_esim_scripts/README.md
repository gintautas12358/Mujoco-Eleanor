# ESIM running commands

## Setup before running

- activate conda vid2e profile
- activate your python venv
- also be advised only one of commands should be active: viewer.capture_frame, viewer.capture_event_prototype, viewer.capture_event

## Displaying different event camera thresholds

- correct fps needs to be set at img/original/seq0/fps.txt
- imgages should be first generated by running ('make sure viewer.capture_frame(camera_id, path=save_path_original) is uncommented'):

- `python3 kuka/PegInHole_camera_raw_imgs.py`

- generate upsampled frames with specified from (takes a considerate amount of time):

- `python rpg_vid2e/upsampling/upsample.py  --input_dir=img/original --output_dir=img/upsampled`
- `python esim/plot_virtual_events_with_diff_param.py`

# Offline

## generate event images

- the frames should be upsampled before running the command.

- `python esim/gen_event_imgs.py`

## generate event images with gpu

- the frames should be upsampled before running the command.

- `python esim/gen_event_imgs_and_events_gpu.py -i img/upsampled -o img/events`

# Online

## generate raw mujoco images

- `python kuka/PegInHole_camera_raw_imgs.py`

## generate subtracted mujoco images

- `python kuka/PegInHole_camera_sub_imgs.py`

## generate mujoco event imgs and events 

- `python kuka/PegInHole_camera_event_imgs_and_events.py`

# Utilities

## create gif

- `convert -delay 20 -loop 0 *.png my.gif`

## run mujoco

- `./../mujoco-2.1.5/bin/simulate`

## plot events

- `python esim/plot_time_space.py`