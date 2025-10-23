#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "aux_functions.h"
#include <time.h>

int main()
{
	while (1) {
		int can_id = rand() % 2; // Frame id 0-7FF 0=AIS 1=GPS
		char *data = malloc(sizeof(char) * 8);
		if (can_id == 0) { // AIS
			data[0] = rand() % 0xFF; // lat
			data[1] = rand() % 0xFF; // lon
			data[2] = rand() % 0xFF; // mmsi bits 0-7
			data[3] = rand() % 0xFF; // mmsi bits 8-15
			data[4] = rand() % 0xFF; // mmsi bits 16-23
			data[5] = rand() % 0x3F; // mmsi bits 24-30
		}

		int result = send_can(can_id, data);

		free(data);
		usleep(100000);
	}

	return 0;
}