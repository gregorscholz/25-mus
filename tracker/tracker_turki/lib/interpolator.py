import numpy as np


DEVIATION_TOLERANCE = 1.5


def _calculate_distances(locations):
    distances = []
    for i in range(0, len(locations) - 1):
        distances.append(
            np.linalg.norm(np.array(locations[i]) - np.array(locations[i + 1]))
        )
    return distances


def _remove_aberrations(locations):
    distances = _calculate_distances(locations)
    avg_distance = np.average(distances)

    for i in range(0, len(distances) - 1):
        deviation_threshold = DEVIATION_TOLERANCE * avg_distance
        if (
            distances[i] > deviation_threshold
            and distances[i + 1] > deviation_threshold
        ):
            print(
                f"removed abberation: ds {distances[i]} {distances[i + 1]} & avg {avg_distance} ==> {locations[i]}, {locations[i + 1]}, {locations[i + 2]}"
            )
            locations[i + 1] = [0, 0]
    return locations


def _get_next_location(locations):
    if locations[0] != [0, 0]:
        return locations[0], 1
    else:
        if locations[1:] == []:
            raise Exception("Ball location ends with unknown!")
        location, step = _get_next_location(locations[1:])
        return location, (step + 1)


def _calculate_one_step(last_known_location, next_known_location, steps):
    distance = [
        (next_known_location[0] - last_known_location[0]),
        (next_known_location[1] - last_known_location[1]),
    ]
    one_step = [(distance[0] / (steps + 1)), (distance[1] / (steps + 1))]
    return one_step


def _interpolate_missing_locations(locations):
    last_known_location = [0, 0]
    for i, location in enumerate(locations):
        if location != [0, 0]:
            last_known_location = location
        else:
            # get next known location and step-number to get there
            next_known_location, steps = _get_next_location(locations[i + 1 :])
            if steps > 10:
                raise Exception("To many missing frames")

            # calculate step size
            one_step = _calculate_one_step(
                last_known_location, next_known_location, steps
            )

            # for this and following missing locations safe new location
            for step in range(0, steps):
                locations[i + step] = [
                    (last_known_location[0] + ((step + 1) * one_step[0])),
                    (last_known_location[1] + ((step + 1) * one_step[1])),
                ]
    return locations


def interpolate(balls_locations, first_frame, last_frame):
    for k, locations in balls_locations.items():
        print(k)
        first_frames = locations[: (first_frame + 1)]
        last_frames = locations[(last_frame + 1) :]
        locations = locations[(first_frame + 1) : (last_frame + 1)]

        # interpolate missing locations
        locations = _interpolate_missing_locations(locations)

        # remove aberration
        locations = _remove_aberrations(locations)

        # interpolate missing locations
        locations = _interpolate_missing_locations(locations)

        locations = first_frames + locations + last_frames
    return balls_locations
