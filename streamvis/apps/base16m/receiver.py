from collections import deque

import numpy as np

from streamvis import receiver

args = receiver.args

HIT_THRESHOLD = 15

peakfinder_buffer = deque(maxlen=args.buffer_size)
last_hit_data = (None, None)
hitrate_buffer_fast = deque(maxlen=50)
hitrate_buffer_slow = deque(maxlen=500)

run_name = ''

run_names = []
nframes = []
bad_frames = []
sat_pix_nframes = []
laser_on_nframes = []
laser_on_hits = []
laser_on_hits_ratio = []
laser_off_nframes = []
laser_off_hits = []
laser_off_hits_ratio = []

stats_table_dict = dict(
    run_names=run_names,
    nframes=nframes,
    bad_frames=bad_frames,
    sat_pix_nframes=sat_pix_nframes,
    laser_on_nframes=laser_on_nframes,
    laser_on_hits=laser_on_hits,
    laser_on_hits_ratio=laser_on_hits_ratio,
    laser_off_nframes=laser_off_nframes,
    laser_off_hits=laser_off_hits,
    laser_off_hits_ratio=laser_off_hits_ratio,
)

sum_nframes = [0]
sum_bad_frames = [0]
sum_sat_pix_nframes = [0]
sum_laser_on_nframes = [0]
sum_laser_on_hits = [0]
sum_laser_on_hits_ratio = [0]
sum_laser_off_nframes = [0]
sum_laser_off_hits = [0]
sum_laser_off_hits_ratio = [0]

sum_stats_table_dict = dict(
    run_names=["Summary"],
    nframes=sum_nframes,
    bad_frames=sum_bad_frames,
    sat_pix_nframes=sum_sat_pix_nframes,
    laser_on_nframes=sum_laser_on_nframes,
    laser_on_hits=sum_laser_on_hits,
    laser_on_hits_ratio=sum_laser_on_hits_ratio,
    laser_off_nframes=sum_laser_off_nframes,
    laser_off_hits=sum_laser_off_hits,
    laser_off_hits_ratio=sum_laser_off_hits_ratio,
)


def on_receive(metadata, image):
    global run_name, last_hit_data
    is_hit = 'number_of_spots' in metadata and metadata['number_of_spots'] > HIT_THRESHOLD

    if 'run_name' in metadata:
        if metadata['run_name'] != run_name:
            current.buffer.clear()
            peakfinder_buffer.clear()
            run_name = metadata['run_name']

            run_names.append(run_name)
            nframes.append(0)
            bad_frames.append(0)
            sat_pix_nframes.append(0)
            laser_on_nframes.append(0)
            laser_on_hits.append(0)
            laser_on_hits_ratio.append(0)
            laser_off_nframes.append(0)
            laser_off_hits.append(0)
            laser_off_hits_ratio.append(0)

        if 'swissmx_x' in metadata and 'swissmx_y' in metadata and \
            'number_of_spots' in metadata and 'frame' in metadata:
            peakfinder_buffer.append(np.array([
                metadata['swissmx_x'], metadata['swissmx_y'], metadata['frame'],
                metadata['number_of_spots'],
            ]))

        nframes[-1] += 1
        sum_nframes[0] += 1
        if 'is_good_frame' in metadata and not metadata['is_good_frame']:
            bad_frames[-1] += 1
            sum_bad_frames[0] += 1
        if 'saturated_pixels' in metadata and metadata['saturated_pixels'] != 0:
            sat_pix_nframes[-1] += 1
            sum_sat_pix_nframes[0] += 1

        if 'laser_on' in metadata:
            if metadata['laser_on']:
                laser_on_nframes[-1] += 1
                sum_laser_on_nframes[0] += 1
                if is_hit:
                    laser_on_hits[-1] += 1
                    sum_laser_on_hits[0] += 1
                laser_on_hits_ratio[-1] = laser_on_hits[-1] / laser_on_nframes[-1]
                sum_laser_on_hits_ratio[0] = sum_laser_on_hits[0] / sum_laser_on_nframes[0]

            else:
                laser_off_nframes[-1] += 1
                sum_laser_off_nframes[0] += 1
                if is_hit:
                    laser_off_hits[-1] += 1
                    sum_laser_off_hits[0] += 1
                laser_off_hits_ratio[-1] = laser_off_hits[-1] / laser_off_nframes[-1]
                sum_laser_off_hits_ratio[0] = sum_laser_off_hits[0] / sum_laser_off_nframes[0]

    if is_hit:
        last_hit_data = (metadata, image)
        hitrate_buffer_fast.append(1)
        hitrate_buffer_slow.append(1)
    else:
        hitrate_buffer_fast.append(0)
        hitrate_buffer_slow.append(0)


current = receiver.Receiver(on_receive=on_receive)
